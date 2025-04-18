import os

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step
from llama_index.utils.workflow import draw_all_possible_flows


class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        return StopEvent(result="Hello, world!")


async def main():
    w = MyWorkflow(timeout=10, verbose=False)
    result = await w.run()
    print(result)
    os.makedirs("workflows", exist_ok=True)
    draw_all_possible_flows(
        w,
        filename="workflows/basic_workflow.html"
    )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
