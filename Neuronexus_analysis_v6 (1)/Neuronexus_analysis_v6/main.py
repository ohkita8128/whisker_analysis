"""
main.py - Neuronexus Analysis v6 メインオーケストレータ

フロー:
1. PLXファイル読み込み
2. LFP Filter GUI → LFP処理パイプライン → LFPプロット
3. Spike Sorting GUI → スパイクソーティング → スパイクプロット
4. Phase Locking GUI → 位相ロック解析 → グランドサマリープロット

使い方:
  python main.py                   # GUI でファイル選択
  python main.py --plx file.plx    # ファイル指定
  python main.py --skip-lfp        # LFP処理をスキップ
  python main.py --skip-spike      # スパイクソーティングをスキップ
"""
import argparse
import os
import sys
import numpy_compat  # NumPy 2.0+ 互換パッチ（neo より先に読み込む）
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Neuronexus Analysis v6")
    parser.add_argument("--plx", type=str, default="", help="PLXファイルパス")
    parser.add_argument("--output", type=str, default="", help="出力ディレクトリ")
    parser.add_argument("--skip-lfp", action="store_true", help="LFP処理をスキップ")
    parser.add_argument("--skip-spike", action="store_true", help="スパイクソーティングをスキップ")
    parser.add_argument("--skip-phase", action="store_true", help="位相ロック解析をスキップ")
    parser.add_argument("--no-wideband", action="store_true", help="Widebandデータを読み込まない")
    args = parser.parse_args()

    print("=" * 60)
    print("  Neuronexus Analysis v6")
    print("  PLX → LFP Filtering → Spike Sorting → Phase Locking")
    print("=" * 60)

    # =========================================================================
    # Step 0: PLXファイル選択 & 読み込み
    # =========================================================================
    plx_file = args.plx
    if not plx_file:
        from get_path import get_path
        plx_file = get_path(mode='file', file_type='plx')

    if not os.path.exists(plx_file):
        print(f"エラー: ファイルが見つかりません: {plx_file}")
        return

    print(f"\n[Step 0] PLX読み込み: {os.path.basename(plx_file)}")
    from data_loader import load_plx
    plx_data = load_plx(plx_file, load_wideband=not args.no_wideband)

    if args.output:
        plx_data.output_dir = args.output
    output_dir = plx_data.output_dir
    basename = plx_data.basename

    # 結果を保持する辞書
    lfp_results = None
    spike_results = None
    phase_results = None

    # =========================================================================
    # Step 1: LFP Filtering (GUI → Pipeline)
    # =========================================================================
    if not args.skip_lfp:
        print(f"\n[Step 1] LFP Filter GUI を起動...")

        from lfp_filter_gui import LfpFilterGUI

        lfp_config_holder = [None]

        def on_lfp_done(config, _plx_data):
            lfp_config_holder[0] = config

        gui = LfpFilterGUI(plx_data=plx_data, on_done=on_lfp_done)
        gui.run()

        lfp_config = lfp_config_holder[0]
        if lfp_config is None:
            print("  LFP処理がキャンセルされました")
            return

        # パイプライン実行
        print(f"\n[Step 1b] LFPパイプライン実行...")
        from lfp_pipeline import run_lfp_pipeline
        lfp_results = run_lfp_pipeline(lfp_config, plx_data)
    else:
        print("\n[Step 1] LFP処理: スキップ")

    # =========================================================================
    # Step 2: Spike Sorting (GUI)
    # =========================================================================
    if not args.skip_spike and plx_data.wideband_raw is not None:
        print(f"\n[Step 2] Spike Sorting GUI を起動...")

        from spike_sort_gui import SpikeSortGUI

        spike_holder = [None]

        def on_spike_done(results):
            spike_holder[0] = results

        gui = SpikeSortGUI(
            wideband_data=plx_data.wideband_raw,
            fs=plx_data.wideband_fs,
            output_dir=output_dir,
            basename=basename,
            on_done=on_spike_done
        )
        gui.run()
        spike_results = spike_holder[0]

        if spike_results:
            # グランドサマリー出力
            from spike_plotting import plot_spike_grand_summary, plot_quality_table
            plot_spike_grand_summary(spike_results, output_dir, basename,
                                     show=True, save=True)
            plot_quality_table(spike_results, output_dir, basename,
                               show=True, save=True)
            print(f"  スパイクソーティング完了: "
                  f"{sum(len(r.units) for r in spike_results.values())} units")
    elif args.skip_spike:
        print("\n[Step 2] スパイクソーティング: スキップ")
    else:
        print("\n[Step 2] Widebandデータなし - スパイクソーティングをスキップ")

    # =========================================================================
    # Step 3: Phase Locking Analysis (GUI)
    # =========================================================================
    if not args.skip_phase and lfp_results is not None:
        print(f"\n[Step 3] Phase Locking GUI を起動...")

        # 条件別マスク
        condition_masks = None
        if all(k in lfp_results for k in ['clean_baseline', 'clean_stim', 'clean_post']):
            condition_masks = {
                'baseline': lfp_results['clean_baseline'],
                'stim': lfp_results['clean_stim'],
                'post': lfp_results['clean_post']
            }

        from phase_gui import PhaseGUI

        phase_holder = [None, None]

        def on_phase_done(pr, cr):
            phase_holder[0] = pr
            phase_holder[1] = cr

        gui = PhaseGUI(
            lfp_cleaned=lfp_results.get('lfp_cleaned'),
            lfp_times=lfp_results.get('lfp_times'),
            fs=lfp_results.get('fs'),
            segment=plx_data.segment,
            spike_results=spike_results,
            stim_times=lfp_results.get('stim_times'),
            condition_masks=condition_masks,
            original_ch_numbers=lfp_results.get('original_ch_numbers', []),
            output_dir=output_dir,
            basename=basename,
            on_done=on_phase_done
        )
        gui.run()
        phase_results = phase_holder[0]
    elif args.skip_phase:
        print("\n[Step 3] 位相ロック解析: スキップ")
    else:
        print("\n[Step 3] LFPデータなし - 位相ロック解析をスキップ")

    # =========================================================================
    # 完了
    # =========================================================================
    print("\n" + "=" * 60)
    print("  Neuronexus Analysis v6 完了!")
    print(f"  出力先: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
