# Module reference

| Module | Responsibility | Main public API |
|---|---|---|
| `tensoralpha.settings` | portable paths and optional credentials | `Settings`, `ConfigurationError` |
| `tensoralpha.data` | market schema and local persistence | `validate_panel`, `ParquetPanelStore` |
| `tensoralpha.data.tushare` | optional provider adapter | `TushareDailySource` |
| `tensoralpha.features` | past-only features and forward labels | `FeaturePipeline`, `add_forward_rank_target` |
| `tensoralpha.models` | sequence Transformer | `TransformerConfig`, `TransformerRanker` |
| `tensoralpha.training` | datasets, training, splits, OOF | `PanelSequenceDataset`, `ModelTrainer`, `PurgedWalkForwardSplit`, `run_purged_oof` |
| `tensoralpha.inference` | strict artifact save/load | `ModelArtifact`, `ModelMetadata` |
| `tensoralpha.evaluation` | Rank IC and showcase aggregates | `rank_ic_by_date`, `summarize_oof`, `derive_oof_showcase_csv` |
| `tensoralpha.strategy` | target-weight construction | `TopNRotationStrategy` |
| `tensoralpha.backtest` | broker and event loop | `SimulatedBroker`, `BacktestEngine` |
| `tensoralpha.paper` | persistent offline account | `PaperTrader`, `PaperState` |
| `tensoralpha.demo` | deterministic synthetic panel | `generate_demo_panel` |
| `tensoralpha.visualization` | deterministic OOF and synthetic backtest SVGs | `render_oof_profile_svg`, `render_backtest_svg` |
| `tensoralpha.release_policy` | pre-publication privacy checks | `scan_repository` |

The supported command surface is `tensoralpha --help`. Internal files may change; the APIs listed above are the intended entry points for the first public release.
