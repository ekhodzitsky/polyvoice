use super::*;
use proptest::prelude::*;

#[test]
fn bare_wav_is_implicit_diarize() {
    let cli = Cli::try_parse_from(["polyvoice", "meeting.wav", "--format", "srt"]).unwrap();
    assert!(cli.command.is_none());
    assert_eq!(cli.diarize.wav.as_deref(), Some(Path::new("meeting.wav")));
    assert_eq!(cli.diarize.format, OutputFormat::Srt);
}

#[test]
fn subcommands_are_not_shadowed_by_default_diarize() {
    assert!(matches!(
        Cli::try_parse_from(["polyvoice", "models", "list"])
            .unwrap()
            .command,
        Some(Command::Models { .. })
    ));
    assert!(matches!(
        Cli::try_parse_from(["polyvoice", "download-models"])
            .unwrap()
            .command,
        Some(Command::DownloadModels { .. })
    ));
    assert!(matches!(
        Cli::try_parse_from(["polyvoice", "completions", "bash"])
            .unwrap()
            .command,
        Some(Command::Completions { .. })
    ));
    assert!(matches!(
        Cli::try_parse_from(["polyvoice", "diarize", "x.wav"])
            .unwrap()
            .command,
        Some(Command::Diarize(_))
    ));
}

proptest! {
    #[test]
    fn profile_from_str_accepts_only_known_names(s in "[a-zA-Z0-9_-]{1,20}") {
        let result = s.parse::<Profile>();
        let lower = s.to_ascii_lowercase();
        if lower == "mobile" || lower == "balanced" || lower == "custom" {
            prop_assert!(result.is_ok());
        } else {
            prop_assert!(result.is_err());
        }
    }

    #[test]
    fn cli_diarize_parses_all_formats(
        profile in "(mobile|balanced|fast)",
        format in "(rttm|json|srt|vtt|txt)",
        threshold in 0.0f32..2.0f32,
        v2 in prop::bool::ANY,
    ) {
        let mut args = vec![
            "polyvoice".to_string(),
            "diarize".to_string(),
            "/tmp/test.wav".to_string(),
            "--profile".to_string(), profile,
            "--format".to_string(), format,
            "--threshold".to_string(), threshold.to_string(),
        ];
        if v2 {
            args.push("--v2".to_string());
        }
        prop_assert!(Cli::try_parse_from(&args).is_ok());
    }

    #[test]
    fn cli_models_info_parses(name in "[a-zA-Z0-9_][a-zA-Z0-9_-]{0,29}") {
        let args = vec![
            "polyvoice".to_string(),
            "models".to_string(),
            "info".to_string(),
            name,
        ];
        prop_assert!(Cli::try_parse_from(&args).is_ok());
    }
}
