from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step
from llama_index.utils.workflow import draw_all_possible_flows
import os


class FirstEvent(Event):
    first_output: str


class SecondEvent(Event):
    second_output: str


class MyWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> FirstEvent:
        print(ev.first_input)
        return FirstEvent(first_output="First step complete.")

    @step
    async def step_two(self, ev: FirstEvent) -> SecondEvent:
        print(ev.first_output)
        return SecondEvent(second_output="Second step complete.")

    @step
    async def step_three(self, ev: SecondEvent) -> StopEvent:
        print(ev.second_output)
        return StopEvent(result="Workflow complete.")


async def main():
    workflow = MyWorkflow(timeout=10, verbose=False)
    result = await workflow.run(first_input="Start the workflow.")
    print(result)

    os.makedirs("workflows", exist_ok=True)

    WORKFLOW_FILE = "workflows/custom_events.html"
    draw_all_possible_flows(workflow, filename=WORKFLOW_FILE)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
