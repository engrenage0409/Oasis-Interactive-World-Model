import sys
import runpy

import interactive_bench

from adaptive_world import AdaptiveWorld

interactive_bench.InteractiveOasis = (
    AdaptiveWorld
)

if __name__ == "__main__":

    try:

        runpy.run_path(
            "benchmark.py",
            run_name="__main__"
        )

    finally:

        model = AdaptiveWorld.last_instance

        if model is not None:

            name = "adaptive"

            if "--name" in sys.argv:
                index = sys.argv.index("--name")
                name = sys.argv[index + 1]

            model.save_report(
                f"benchmarks/{name}/adaptive_report.json"
            )
