import asyncio
import random
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step, Context


class StepTwoEvent(Event):
    query: str


class ParallelFlow(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> StepTwoEvent:
        ctx.send_event(StepTwoEvent(query="Query 1"))
        ctx.send_event(StepTwoEvent(query="Query 2"))
        ctx.send_event(StepTwoEvent(query="Query 3"))

        return StepTwoEvent(query="Dummy query to satisfy return")


    @step(num_workers=4)
    async def step_two(self, ctx: Context, ev: StepTwoEvent) -> StopEvent:
        print(f"Running slow query: {ev.query}")
        await asyncio.sleep(random.randint(1, 5))

        return StopEvent(result=ev.query)


async def main():
    parallel_workflow = ParallelFlow(timeout=10, verbose=False)

    result = await parallel_workflow.run()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
