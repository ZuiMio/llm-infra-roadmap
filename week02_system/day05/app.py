from metrics import summarize_step_times


def main():
    """Run the training step time summary example."""
    step_times = [0.82, 0.91, 1.12, 0.88, 2.05]

    summary = summarize_step_times(step_times)

    print("Training step time summary")
    print(f"step_count: {summary['count']}")
    print(f"mean_step_time_seconds: {summary['mean_seconds']:.3f}")
    print(f"max_step_time_seconds: {summary['max_seconds']:.3f}")


if __name__ == "__main__":
    main()