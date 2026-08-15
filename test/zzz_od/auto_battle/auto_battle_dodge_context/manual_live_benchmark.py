from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import threading
import tracemalloc
from collections.abc import Callable
from concurrent.futures import TimeoutError
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter_ns, sleep, thread_time_ns, time
from unittest.mock import MagicMock

import numpy as np
import psutil
from scipy.signal import correlate

from one_dragon.utils.log_utils import LoggerConfig, configure_logger, log
from zzz_od.auto_battle.auto_battle_dodge_context import (
    AudioTemplateEnum,
    AutoBattleDodgeContext,
)
from zzz_od.backend.backend_context import ZzzBackendContext
from zzz_od.context.zzz_context import ZContext


@dataclass(slots=True)
class AudioSample:
    """一次实战音频窗口的新旧算法结果。"""

    new_elapsed_ms: float
    legacy_elapsed_ms: float
    scores: np.ndarray
    legacy_score: float
    elapsed_seconds: float


class LiveBenchmark:
    """在真实应用运行期间比较新三模板算法和旧单模板算法。"""

    def __init__(
        self,
        progress_path: Path,
        report_path: Path,
        events_path: Path | None = None,
    ) -> None:
        self.progress_path: Path = progress_path
        self.report_path: Path = report_path
        self.events_path: Path | None = events_path
        self.samples: list[AudioSample] = []
        self.events: list[AudioTemplateEnum] = []
        self.battle_boundaries: list[float] = []
        self.resource_audio_samples: list[np.ndarray] = []
        self._started_at: float = monotonic()
        self._legacy_templates: dict[int, np.ndarray] = {}
        self._resource_context: AutoBattleDodgeContext | None = None
        self._lock = threading.Lock()
        self._thread_state = threading.local()
        self._original_get_max_corr = AutoBattleDodgeContext.get_max_corr
        self._original_check_dodge_audio = AutoBattleDodgeContext.check_dodge_audio

    def emit(self, message: str) -> None:
        """同时写入终端和独立进度文件。"""
        print(message, flush=True)
        with self.progress_path.open('a', encoding='utf-8') as progress_file:
            progress_file.write(f'{message}\n')

    def emit_event(self, event_data: dict[str, object]) -> None:
        """按 JSON Lines 格式保存不丢精度的原始事件。"""
        if self.events_path is None:
            return
        with self.events_path.open('a', encoding='utf-8') as events_file:
            events_file.write(f'{json.dumps(event_data, ensure_ascii=False)}\n')

    def record_battle_boundary(self, completed_runs: int) -> None:
        """记录计划完成次数变化，用它作为每轮战斗的结束边界。"""
        elapsed_seconds = monotonic() - self._started_at
        self.battle_boundaries.append(elapsed_seconds)
        self.emit(f'[分轮] 第 {completed_runs} 轮完成，采样时间={elapsed_seconds:.3f}s')
        self.emit_event({
            'type': 'battle_completed',
            'completed_runs': completed_runs,
            'elapsed_seconds': elapsed_seconds,
            'timestamp': time(),
        })

    def install(self) -> None:
        """只在当前测试进程中安装计时和计数钩子。"""
        benchmark = self

        def measured_get_max_corr(
            context: AutoBattleDodgeContext,
            audio: np.ndarray,
        ) -> np.ndarray:
            # 新算法计时只包住正式实现，不包含旧算法和统计开销。
            started = perf_counter_ns()
            scores = benchmark._original_get_max_corr(context, audio)
            new_elapsed_ms = (perf_counter_ns() - started) / 1_000_000

            started = perf_counter_ns()
            legacy_score = benchmark._legacy_get_max_corr(context, audio)
            legacy_elapsed_ms = (perf_counter_ns() - started) / 1_000_000

            sample = AudioSample(
                new_elapsed_ms=new_elapsed_ms,
                legacy_elapsed_ms=legacy_elapsed_ms,
                scores=scores.copy(),
                legacy_score=legacy_score,
                elapsed_seconds=monotonic() - benchmark._started_at,
            )
            benchmark._thread_state.latest_sample = sample
            with benchmark._lock:
                benchmark.samples.append(sample)
                sample_count = len(benchmark.samples)
                benchmark._resource_context = context
                if (
                    sample_count % 50 == 0
                    and len(benchmark.resource_audio_samples) < 256
                ):
                    benchmark.resource_audio_samples.append(audio.copy())
            if sample_count % 250 == 0:
                benchmark.emit(
                    f'[采样] 窗口={sample_count} '
                    f'新三模板={new_elapsed_ms:.3f}ms '
                    f'旧单模板={legacy_elapsed_ms:.3f}ms '
                    f'当前最大相关={float(np.max(scores)):.3f}'
                )
            return scores

        def measured_check_dodge_audio(
            context: AutoBattleDodgeContext,
            screenshot_time: float,
        ) -> AudioTemplateEnum | bool:
            result = benchmark._original_check_dodge_audio(context, screenshot_time)
            if result is False:
                return False

            event = AudioTemplateEnum(result)
            with benchmark._lock:
                benchmark.events.append(event)
            sample: AudioSample = benchmark._thread_state.latest_sample
            benchmark.emit_event({
                'type': 'audio_hit',
                'elapsed_seconds': sample.elapsed_seconds,
                'timestamp': time(),
                'result': event.name,
                'scores': sample.scores.tolist(),
                'legacy_score': sample.legacy_score,
                'new_elapsed_ms': sample.new_elapsed_ms,
                'legacy_elapsed_ms': sample.legacy_elapsed_ms,
            })
            benchmark.emit(
                f'[命中] {event.name} '
                f'scores={np.round(sample.scores, 3).tolist()} '
                f'旧模板={sample.legacy_score:.3f} '
                f'新={sample.new_elapsed_ms:.3f}ms '
                f'旧={sample.legacy_elapsed_ms:.3f}ms'
            )
            return event

        AutoBattleDodgeContext.get_max_corr = measured_get_max_corr
        AutoBattleDodgeContext.check_dodge_audio = measured_check_dodge_audio

    def uninstall(self) -> None:
        """恢复当前进程中的正式实现。"""
        AutoBattleDodgeContext.get_max_corr = self._original_get_max_corr
        AutoBattleDodgeContext.check_dodge_audio = self._original_check_dodge_audio

    def _legacy_get_max_corr(
        self,
        context: AutoBattleDodgeContext,
        audio: np.ndarray,
    ) -> float:
        """执行升级前的单模板相关系数算法。"""
        context_id = id(context)
        template = self._legacy_templates.get(context_id)
        if template is None:
            template_path = Path('assets/template/dodge_audio/template_1.wav')
            template = context._get_filter_wave(context._load_audio_template(template_path))
            self._legacy_templates[context_id] = template

        filtered_audio = context._get_filter_wave(audio)
        scaled_template = context._standardize_wave(template)
        scaled_audio = context._standardize_wave(filtered_audio)
        if scaled_template.size > scaled_audio.size:
            correlation = correlate(
                scaled_template,
                scaled_audio,
                mode='same',
                method='fft',
            )
            denominator = scaled_template.size
        else:
            correlation = correlate(
                scaled_audio,
                scaled_template,
                mode='same',
                method='fft',
            )
            denominator = scaled_audio.size
        return float(np.max(correlation) / denominator)

    def build_report(self) -> dict[str, object]:
        """生成便于人工复核和后续对比的汇总数据。"""
        if not self.samples:
            return {'sample_count': 0, 'event_count': 0}

        new_times = np.asarray([sample.new_elapsed_ms for sample in self.samples])
        legacy_times = np.asarray([sample.legacy_elapsed_ms for sample in self.samples])
        scores = np.stack([sample.scores for sample in self.samples])
        legacy_scores = np.asarray([sample.legacy_score for sample in self.samples])
        new_hit_mask = np.max(scores, axis=1) > 0.1
        legacy_hit_mask = legacy_scores > 0.1
        type_indexes = np.argmax(scores, axis=1)

        hit_margins = np.empty(0, dtype=np.float64)
        if np.any(new_hit_mask):
            sorted_hit_scores = np.sort(scores[new_hit_mask], axis=1)
            hit_margins = sorted_hit_scores[:, -1] - sorted_hit_scores[:, -2]

        return {
            'sample_count': len(self.samples),
            'new_elapsed_ms': self._describe_times(new_times),
            'legacy_elapsed_ms': self._describe_times(legacy_times),
            'new_hit_count': int(np.sum(new_hit_mask)),
            'legacy_hit_count': int(np.sum(legacy_hit_mask)),
            'threshold_hit_count': {
                event.name: int(np.sum(scores[:, event.value - 1] > 0.1))
                for event in AudioTemplateEnum
            },
            'additional_hit_count': int(np.sum(new_hit_mask & ~legacy_hit_mask)),
            'classified_hit_count': {
                event.name: int(
                    np.sum(new_hit_mask & (type_indexes == event.value - 1))
                )
                for event in AudioTemplateEnum
            },
            'returned_event_count': {
                event.name: self.events.count(event)
                for event in AudioTemplateEnum
            },
            'hit_margin': {
                'minimum': float(np.min(hit_margins)) if hit_margins.size else None,
                'mean': float(np.mean(hit_margins)) if hit_margins.size else None,
            },
            'resource_usage': self._build_resource_usage_report(),
            'per_battle': self._build_per_battle_report(),
        }

    def _build_resource_usage_report(self) -> dict[str, object]:
        """用同一批真实录音窗口分别测量两种算法的 CPU 和内存。"""
        context = self._resource_context
        audio_samples = self.resource_audio_samples
        if context is None or not audio_samples:
            return {'sample_count': 0}

        # 先预热模板缓存和 FFT 路径，避免把首次初始化记入稳定运行资源消耗。
        for audio in audio_samples[:8]:
            self._original_get_max_corr(context, audio)
            self._legacy_get_max_corr(context, audio)

        return {
            'sample_count': len(audio_samples),
            'method': (
                '同一批真实录音窗口分别重放；CPU 使用当前采样线程时间，'
                '内存使用 tracemalloc 峰值及进程 RSS 轮询峰值'
            ),
            'new_three_template': self._measure_resource_usage(
                lambda audio: self._original_get_max_corr(context, audio),
                audio_samples,
            ),
            'legacy_single_template': self._measure_resource_usage(
                lambda audio: self._legacy_get_max_corr(context, audio),
                audio_samples,
            ),
        }

    @staticmethod
    def _measure_resource_usage(
        operation: Callable[[np.ndarray], np.ndarray | float],
        audio_samples: list[np.ndarray],
    ) -> dict[str, float]:
        """分开测 CPU 与内存，避免内存跟踪污染 CPU 结果。"""
        process = psutil.Process()

        wall_started = perf_counter_ns()
        cpu_started = thread_time_ns()
        for audio in audio_samples:
            operation(audio)
        cpu_seconds = (thread_time_ns() - cpu_started) / 1_000_000_000
        wall_seconds = (perf_counter_ns() - wall_started) / 1_000_000_000

        gc.collect()
        tracemalloc.start()
        traced_before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        rss_before = process.memory_info().rss
        rss_peak = rss_before
        for audio in audio_samples:
            operation(audio)
            rss_peak = max(rss_peak, process.memory_info().rss)
        rss_after = process.memory_info().rss
        _, traced_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        sample_count = len(audio_samples)
        mib = 1024 * 1024
        return {
            'cpu_total_ms': cpu_seconds * 1000,
            'cpu_mean_ms_per_window': cpu_seconds * 1000 / sample_count,
            'single_thread_cpu_percent': (
                cpu_seconds / wall_seconds * 100 if wall_seconds > 0 else 0.0
            ),
            'rss_before_mib': rss_before / mib,
            'rss_peak_mib': rss_peak / mib,
            'rss_after_mib': rss_after / mib,
            'rss_peak_increment_mib': (rss_peak - rss_before) / mib,
            'traced_peak_increment_mib': (traced_peak - traced_before) / mib,
        }

    def _build_per_battle_report(self) -> list[dict[str, object]]:
        """按照计划完成次数记录的边界汇总每轮结果。"""
        if not self.battle_boundaries:
            return []
        reports: list[dict[str, object]] = []
        battle_start = 0.0
        for battle_idx, battle_end in enumerate(self.battle_boundaries, start=1):
            battle_samples = [
                sample
                for sample in self.samples
                if battle_start <= sample.elapsed_seconds < battle_end
            ]
            if battle_samples:
                scores = np.stack([sample.scores for sample in battle_samples])
                legacy_scores = np.asarray([
                    sample.legacy_score for sample in battle_samples
                ])
                type_indexes = np.argmax(scores, axis=1)
                max_scores = np.max(scores, axis=1)
                reports.append({
                    'battle': battle_idx,
                    'start_seconds': battle_start,
                    'end_seconds': battle_end,
                    'sample_count': len(battle_samples),
                    'threshold_hit_count': {
                        event.name: int(np.sum(scores[:, event.value - 1] > 0.1))
                        for event in AudioTemplateEnum
                    } | {
                        'LEGACY_TEMPLATE': int(np.sum(legacy_scores > 0.1)),
                    },
                    'argmax_hit_count': {
                        event.name: int(np.sum(
                            (max_scores > 0.1)
                            & (type_indexes == event.value - 1)
                        ))
                        for event in AudioTemplateEnum
                    } | {
                        'LEGACY_TEMPLATE': int(np.sum(legacy_scores > 0.1)),
                    },
                })
            battle_start = battle_end
        return reports

    def write_report(self) -> None:
        """写出 JSON 报告并把摘要同步到进度文件。"""
        report = self.build_report()
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        self.emit(f'[汇总] {json.dumps(report, ensure_ascii=False)}')

    @staticmethod
    def _describe_times(times: np.ndarray) -> dict[str, float]:
        """返回耗时数组的常用统计量。"""
        return {
            'mean': float(np.mean(times)),
            'p50': float(np.percentile(times, 50)),
            'p95': float(np.percentile(times, 95)),
            'maximum': float(np.max(times)),
        }


def _read_first_plan_run_times(config_path: Path | None) -> int | None:
    """读取体力计划第一项的完成次数，不修改配置。"""
    if config_path is None or not config_path.is_file():
        return None
    match = re.search(
        r'^\s*run_times:\s*(\d+)\s*$',
        config_path.read_text(encoding='utf-8'),
        flags=re.MULTILINE,
    )
    return int(match.group(1)) if match is not None else None


def parse_arguments() -> argparse.Namespace:
    """解析手动实战测试参数。"""
    parser = argparse.ArgumentParser(description='实战比较新旧声音闪避识别算法')
    parser.add_argument('--app-id', default='charge_plan')
    parser.add_argument('--max-seconds', type=int, default=900)
    parser.add_argument('--audio-only', action='store_true')
    parser.add_argument('--stop-after-runs', type=int)
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--events', type=Path)
    parser.add_argument('--plan-config', type=Path)
    return parser.parse_args()


def main() -> None:
    """运行当前实例配置的应用并采集实战音频。"""
    args = parse_arguments()
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.progress.write_text('', encoding='utf-8')
    if args.events is not None:
        args.events.parent.mkdir(parents=True, exist_ok=True)
        args.events.write_text('', encoding='utf-8')

    # 单独使用测试日志，避免正式日志被其他一条龙进程占用时反复轮转失败。
    configure_logger(
        log,
        LoggerConfig(
            level=logging.INFO,
            log_file_path='audio_live_benchmark.log',
            add_console_handler=False,
        ),
    )

    benchmark = LiveBenchmark(args.progress, args.report, args.events)
    benchmark.install()
    if args.audio_only:
        dodge_context = AutoBattleDodgeContext(MagicMock())
        try:
            benchmark.emit('[启动] 正在加载声音模板……')
            dodge_context.init_audio_template()
            dodge_context.start_context_async()
            benchmark.emit(
                f'[启动] 音频直采已开始，将持续 {args.max_seconds} 秒；请进入实战。'
            )
            deadline = monotonic() + args.max_seconds
            previous_run_times = _read_first_plan_run_times(args.plan_config)
            next_config_check = monotonic()
            reached_target_runs = False
            while monotonic() < deadline:
                dodge_context.check_dodge_audio(monotonic())
                if args.plan_config is not None and monotonic() >= next_config_check:
                    current_run_times = _read_first_plan_run_times(args.plan_config)
                    if (
                        current_run_times is not None
                        and previous_run_times is not None
                        and current_run_times > previous_run_times
                    ):
                        for completed_runs in range(
                            previous_run_times + 1,
                            current_run_times + 1,
                        ):
                            benchmark.record_battle_boundary(completed_runs)
                        if (
                            args.stop_after_runs is not None
                            and len(benchmark.battle_boundaries)
                            >= args.stop_after_runs
                        ):
                            benchmark.emit(
                                f'[结束] 已记录 {args.stop_after_runs} 轮完成边界。'
                            )
                            reached_target_runs = True
                    if current_run_times is not None:
                        previous_run_times = current_run_times
                    next_config_check = monotonic() + 0.1
                if reached_target_runs:
                    break
                sleep(0.01)
            if not reached_target_runs:
                benchmark.emit('[结束] 音频直采达到时间上限。')
        except Exception as error:
            benchmark.emit(f'[失败] {type(error).__name__}: {error}')
            raise
        finally:
            dodge_context.stop_context()
            dodge_context.after_app_shutdown()
            benchmark.uninstall()
            benchmark.write_report()
        return

    ctx = ZContext()
    backend = ZzzBackendContext(ctx)
    try:
        # 当前 ppocrv6 资产不完整时，测试进程临时使用本地已有的 ppocrv5，不写回用户配置。
        ocr_model_dir = Path('assets/models/onnx_ocr') / ctx.model_config.ocr
        if not (ocr_model_dir / 'det.onnx').is_file():
            fallback_ocr_dir = Path('assets/models/onnx_ocr/ppocrv5')
            if (fallback_ocr_dir / 'det.onnx').is_file():
                ctx.model_config.data['ocr'] = 'ppocrv5'
                benchmark.emit('[启动] ppocrv6 资产不完整，测试临时使用本地 ppocrv5。')
        benchmark.emit('[启动] 正在初始化一条龙上下文……')
        ctx.init()
        benchmark.emit(
            f'[启动] 当前实例={ctx.current_instance_idx} '
            f'独立应用={ctx.standalone_app_config.active_app_id}'
        )
        ok, future = backend.run_standalone_app(
            'audio_live_benchmark',
            app_id=args.app_id,
        )
        if not ok or future is None:
            raise RuntimeError('应用未能启动：当前已有其他任务运行')

        benchmark.emit(f'[启动] {args.app_id} 已开始，正在采集实战音频……')
        try:
            result = future.result(timeout=args.max_seconds)
            benchmark.emit(f'[结束] 应用结果: {result}')
        except TimeoutError:
            benchmark.emit(f'[结束] 达到 {args.max_seconds} 秒上限，正在停止应用。')
            benchmark.emit(f'[结束] 停止结果: {backend.stop()}')
            future.result(timeout=60)
    except Exception as error:
        benchmark.emit(f'[失败] {type(error).__name__}: {error}')
        raise
    finally:
        try:
            ctx.after_app_shutdown()
        finally:
            benchmark.uninstall()
            benchmark.write_report()


if __name__ == '__main__':
    main()
