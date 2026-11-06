"""Matched-concurrency, warmed native-API measurements with raw request records."""

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import tinker
from tinker import types

from evals.pricing import TINKER_PRICES

OUT = Path("results/controlled")
PREFIX = "auto1105r2"
MODELS = {
    "qwen35-4b": "Qwen/Qwen3.5-4B",
    "qwen35-9b": "Qwen/Qwen3.5-9B",
    "nemotron35-lightning": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
}
CONCURRENCIES = [1, 8, 16, 32, 64]


def elapsed_clock():
    # macOS time.monotonic() excludes host sleep; CLOCK_MONOTONIC includes it.
    clock = getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC)
    return time.clock_gettime(clock)


def reject_suspension(started, awake_started):
    suspended = elapsed_clock() - started - (time.monotonic() - awake_started)
    if abs(suspended) > 2:
        raise RuntimeError(f"Host suspended during benchmark: clock difference {suspended:.3f}s")


def now():
    return datetime.now(UTC).isoformat()


def emit(event, **values):
    sys.stdout.write(json.dumps(dict(event=event, timestamp=now(), **values)) + "\n")
    sys.stdout.flush()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def modal_cli(*args):
    process = subprocess.run(
        ["modal", *args, "--env", "main"], text=True, capture_output=True, check=True
    )
    return json.loads(process.stdout) if process.stdout.strip() else None


def validate_autoscaling(info, maximum):
    assert info['max_containers'] == maximum and info['min_containers'] in (None, 0), info


def cleanup(model=None):
    apps = modal_cli("app", "list", "--json")
    stopped = []
    for app in apps:
        name = app["description"]
        wanted = name == f"spindle-lora-{PREFIX}-{model}" if model else (
            name == f"spindle-{PREFIX}" or name.startswith(f"spindle-lora-{PREFIX}-")
        )
        if wanted and app["state"] != "stopped":
            modal_cli("app", "stop", app["app_id"], "--yes")
            stopped.append(dict(name=name, app_id=app["app_id"]))
    emit("cleanup", model=model, stopped=stopped)


def workload():
    return json.loads(Path('reference/controlled/workload-manifest.json').read_text())


def prepare(tokenizer, rows):
    return [dict(
        key=row["key"],
        input_ids=tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            add_generation_prompt=True, enable_thinking=False, tokenize=True,
        )["input_ids"],
    ) for row in rows]


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


async def sample_batch(sampler, prompts, params, num_samples, concurrency, context):
    pending = iter(enumerate(prompts))
    responses = []
    started_utc = now()
    started = elapsed_clock()
    awake_started = time.monotonic()
    finished = []

    async def worker():
        for index, prompt in pending:
            dispatched = elapsed_clock()
            dispatched_utc = now()
            response = await asyncio.wait_for(sampler.sample_async(
                prompt=types.ModelInput.from_ints(prompt["input_ids"]),
                num_samples=num_samples, sampling_params=params,
            ), timeout=1800)
            completed = elapsed_clock()
            responses.append((index, prompt, response, dispatched, completed, dispatched_utc, now()))
            finished.append(completed)
            if len(responses) == 1 or len(responses) % 16 == 0 or len(responses) == len(prompts):
                emit("batch_progress", **context, completed=len(responses), total=len(prompts),
                     elapsed_s=round(completed - started, 3))

    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(prompts)))))
    ended = elapsed_clock()
    reject_suspension(started, awake_started)
    finished_utc = now()
    records = []
    for index, prompt, response, dispatched, completed, dispatched_utc, completed_utc in responses:
        assert len(response.sequences) == num_samples
        sequences = [dict(sequence_id=s.sequence_id, token_ids=s.tokens,
                          stop_reason=str(s.stop_reason)) for s in response.sequences]
        assert all(sequence["token_ids"] for sequence in sequences)
        records.append(dict(
            **context, index=index, key=prompt["key"],
            input_sha256=hashlib.sha256(json.dumps(prompt["input_ids"]).encode()).hexdigest(),
            prompt_tokens=len(prompt["input_ids"]),
            cached_prompt_tokens=response.prompt_cache_hit_tokens,
            sequences=sequences, output_tokens=sum(len(s["token_ids"]) for s in sequences),
            dispatched_at=dispatched_utc, completed_at=completed_utc,
            dispatch_offset_s=dispatched-started, complete_offset_s=completed-started,
            request_latency_s=completed-dispatched,
            first_token_at=None, first_token_unavailable_reason="native API returns complete requests",
        ))
    records.sort(key=lambda record: record["index"])
    assert len(records) == len(prompts)
    latencies = [record["request_latency_s"] for record in records]
    generated = sum(record["output_tokens"] for record in records)
    summary = dict(
        **context, started_at=started_utc, finished_at=finished_utc,
        request_count=len(records), sequence_count=len(records)*num_samples,
        request_concurrency=concurrency, max_offered_sequence_concurrency=concurrency*num_samples,
        sample_tokens=generated, prompt_tokens=sum(record["prompt_tokens"] for record in records),
        wall_s=ended-started, last_request_completion_s=max(finished)-started,
        output_tokens_per_second=generated/(ended-started),
        first_completed_request_s=min(finished)-started,
        request_latency_p50_s=statistics.median(latencies),
        request_latency_p95_s=percentile(latencies, 0.95),
        request_latency_p99_s=percentile(latencies, 0.99),
        ttft_s=None, timer="sleep-inclusive monotonic", measurement="warm native-API batch; pretokenized inputs; no client decoding/scoring",
    )
    return summary, records


def persist(summary, records, model, backend, phase):
    directory = OUT / model / backend
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{phase}-requests.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    if backend == "tinker":
        prefill, output = TINKER_PRICES[MODELS[model]]
        samples = summary["sequence_count"] // summary["request_count"]
        cached = sum(record["cached_prompt_tokens"] + record["prompt_tokens"]*(samples-1)
                     for record in records)
        uncached = sum(record["prompt_tokens"]-record["cached_prompt_tokens"] for record in records)
        assert uncached >= 0
        summary["published_rate_cost_estimate_usd"] = (
            uncached*prefill + cached*prefill*0.2 + summary["sample_tokens"]*output
        ) / 1e6
        summary["cached_billed_prompt_tokens"] = cached
    save(directory / f"{phase}-summary.json", summary)
    with (OUT / "phase-summaries.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")
    emit("phase_summary", **summary)


async def run_model(short, data):
    model = MODELS[short]
    clients = {}
    prepared = {}
    for backend in ["spindle"]:
        service = await asyncio.to_thread(tinker.ServiceClient,
            base_url=os.environ["SPINDLE_BASE_URL"] if backend == "spindle" else None,
            api_key=os.environ["SPINDLE_API_KEY"] if backend == "spindle" else os.environ["TINKER_API_KEY"],
            user_metadata={"orx_run_id": os.environ["ORX_RUN_ID"], "controlled_model": short,
                           "controlled_backend": backend})
        sampler = await asyncio.to_thread(service.create_sampling_client, base_model=model)
        tokenizer = await asyncio.to_thread(sampler.get_tokenizer)
        clients[backend] = (service, sampler)
        prepared[backend] = {task: prepare(tokenizer, spec["rows"]) for task, spec in data.items()}
        emit("session_ready", model=model, backend=backend, console_url=service.get_console_url())
    reference = json.loads(Path(f'reference/controlled/{short}-prompt-token-manifest.json').read_text())
    assert prepared['spindle'] == reference, 'Prompt tokens differ from the fixed-worker/Tinker runs'
    save(OUT / short / "prompt-token-manifest.json", prepared["spindle"])

    async def warm(backend, phase, concurrency=32):
        sampler = clients[backend][1]
        probe = prepared[backend]["ifeval"][:32]
        summary, records = await sample_batch(sampler, probe,
            types.SamplingParams(max_tokens=16, temperature=0.0, seed=1729), 1, concurrency,
            dict(model=model, backend=backend, phase=phase, measurement_kind="warmup"))
        await asyncio.to_thread(persist, summary, records, short, backend, phase)

    await asyncio.gather(*(warm(backend, "initial-warmup") for backend in clients))
    server = f"spindle-lora-{PREFIX}-{short}/Server"
    info = await asyncio.to_thread(modal_cli, "server", "info", server, "--json")
    maximum = 4 if short == 'nemotron35-lightning' else 8
    validate_autoscaling(info, maximum)
    save(OUT / short / "server-info.json", info)
    emit("default_autoscaling_verified", model=model, object_id=info["object_id"],
         gpus=info["gpus"], min_containers=0, max_containers=maximum)
    try:
        for repeat in range(2):
            tasks = ["ifeval", "math500"] if repeat == 0 else ["math500", "ifeval"]
            for task in tasks:
                spec = data[task]
                concurrencies = CONCURRENCIES if repeat == 0 else list(reversed(CONCURRENCIES))
                for concurrency in concurrencies:
                    phase = f"{task}-c{concurrency}-r{repeat}"
                    await asyncio.gather(*(warm(backend, phase+"-probe", min(concurrency, 32))
                                           for backend in clients))
                    emit("matched_phase_start", model=model, task=task, concurrency=concurrency,
                         repeat=repeat, prompts=len(spec["rows"]), num_samples=spec["num_samples"])

                    async def measure(backend):
                        params = types.SamplingParams(max_tokens=spec["max_tokens"],
                            temperature=spec["temperature"], seed=1729+repeat)
                        summary, records = await sample_batch(clients[backend][1], prepared[backend][task],
                            params, spec["num_samples"], concurrency,
                            dict(model=model, backend=backend, task=task, repeat=repeat,
                                 phase=phase, measurement_kind="measured"))
                        return summary, records

                    results = await asyncio.gather(*(measure(backend) for backend in clients))
                    await asyncio.gather(*(asyncio.to_thread(persist, summary, records,
                        short, summary["backend"], phase) for summary, records in results))
                    summaries = [summary for summary, records in results]
                    stats = await asyncio.to_thread(modal_cli, "server", "stats", info["object_id"],
                        "--since", next(s["started_at"] for s in summaries if s["backend"] == "spindle"),
                        "--json")
                    save(OUT / short / f"{phase}-server-stats.json", stats)
                    assert stats["container_error_count"] == 0, stats
                    metrics = await asyncio.to_thread(modal_cli, 'server', 'metrics', info['object_id'],
                        '--since', summaries[0]['started_at'], '--group', 'containers', '--group', 'gpu',
                        '--bucket-size', '1m', '--json')
                    save(OUT / short / f'{phase}-worker-metrics.json', metrics)
        emit("model_complete", model=model)
    finally:
        await asyncio.to_thread(cleanup, short)


async def heartbeat():
    while True:
        started, awake_started = elapsed_clock(), time.monotonic()
        await asyncio.sleep(15)
        reject_suspension(started, awake_started)
        emit("controller_heartbeat")
        apps = await asyncio.to_thread(modal_cli, 'app', 'list', '--json')
        snapshot = dict(timestamp=now(), apps=[dict(name=a['description'], app_id=a['app_id'],
            state=a['state'], tasks=int(a['tasks'])) for a in apps
            if a['description'].startswith(f'spindle-lora-{PREFIX}-')])
        with (OUT/'worker-timeline.jsonl').open('a') as handle:
            handle.write(json.dumps(snapshot)+'\n')
        emit('worker_snapshot', **{k:v for k,v in snapshot.items() if k!='timestamp'})


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = await asyncio.to_thread(workload)
    save(OUT / "workload-manifest.json", data)
    emit("protocol", concurrency=CONCURRENCIES, repeats=2, ifeval_prompts=128,
         math_prompts=64, math_samples=4, spindle_min_workers=0, spindle_max_workers=[8,8,4],
         spindle_default_autoscaling=True, spindle_target_concurrency=[32,16,64],
         pool_idle_timeout_s=300, session_idle_timeout_s=300, reused_baseline_run_ids=['822fee2a-efb1-4112-865a-be28c196867e',
         'c527d650-9968-4721-8a38-d362e567b652'], thinking=False,
         first_token_timestamps_available=False, run_id=os.environ["ORX_RUN_ID"])
    async with asyncio.timeout(3600), asyncio.TaskGroup() as group:
        pulse = group.create_task(heartbeat())
        jobs = [group.create_task(run_model(model, data)) for model in MODELS]
        await asyncio.gather(*jobs)
        pulse.cancel()
    emit("suite_complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup-only", action="store_true")
    args = parser.parse_args()
    if args.cleanup_only:
        cleanup()
    else:
        asyncio.run(main())
