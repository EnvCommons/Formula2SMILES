from openai import OpenAI
from openreward import AsyncOpenReward
import asyncio
import json

async def test_function():
    or_client = AsyncOpenReward()
    oai_client = OpenAI()
    MODEL_NAME = "gpt-5.2"

    environment = or_client.environments.get(name="local/Formula2SMILES", base_url="http://localhost:8080")
    tasks = await environment.list_tasks(split="test")
    tools = await environment.list_tools(format="openai")
    example_task = tasks[0]

    async with environment.session(task=example_task) as session:
        prompt = await session.get_prompt()
        input_list = [{"role": "user", "content": prompt[0].text}]
        finished = False
        print(input_list)

        while not finished:
            response = oai_client.responses.create(
                model=MODEL_NAME,
                tools=tools,
                input=input_list
            )

            print(response.output)

            input_list += response.output

            for item in response.output:
                if item.type == "function_call":

                    tool_result = await session.call_tool(item.name, json.loads(str(item.arguments)))
                    reward = tool_result.reward
                    finished = tool_result.finished

                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps({
                            "result": tool_result.blocks[0].text
                        })
                    })

                    print(input_list[-1])

                    if tool_result.finished:
                        finished = True
                        break

if __name__ == "__main__":
    asyncio.run(test_function())
