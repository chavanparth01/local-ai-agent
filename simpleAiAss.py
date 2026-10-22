from openai import OpenAI
from dotenv import load_dotenv
from Tools.tools import run_tool
from Tools.tools_schema import Tools
from Utils.utilities import log
import os
import json

load_dotenv()

EPOCHS = 10

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

# !3 Roles : user, assistant, system

# !Implementing a simple chat loop to interact with the model

# !Adding context of the entire chat to messages = []

# !Adding tool calling functionality to the chat loop

# !Adding more tools like Weather API

messages = []

messages.append({
    "role" : "system",
    "content" : (
        "you are a helpful assistant."
        "answer concisely, in at most 2 short sentences, unless the user asks for more details."
        "call the getCurrentTime tool when asked for the current time."
        "call the get_weather tool when asked for the current weather in a city."
        "If a tool returns an error, explain the error to the user"
        "in simple language. Do not invent missing weather information."
        "call the get_user_city tool when user's city is needed"
        "When the user asks for the latest, newest, current, or most recent "
        "information, you MUST call web_search before answering. "
        "Construct a query that asks for the latest result, without assuming "
        "an outdated year or winner. When search results contain multiple years, "
        "compare all returned results and identify the newest confirmed result. "
        "Do not simply use the first search result. Base your answer on the "
        "search results and include the source URLs. If the results are "
        "ambiguous or conflicting, say so rather than guessing."
    )
    
})


while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit", "bye"]: 
        break

    messages.append({
            "role" : "user",
            "content" : user_input,
    })

    # ?First API call to the model to get the response and check if it wants to call any tool
    # ?This call will also be made inside the while loop until the model has no more tool calls to make and it will return the final response to the user
    # response = client.chat.completions.create(
    #     model = os.getenv("MODEL"),
    #     messages=messages,
    #     tools = Tools,
    # )

    # model_response = response.choices[0].message
    # print(json.dumps(response))

    # ?Check if model wants to call any tool and if yes, call the tool and add the tool response to the messages array so that the model can use it in the next turn
    for iteration in range(EPOCHS): 
        response = client.chat.completions.create(
            model = os.getenv("MODEL"),
            messages = messages,
            tools = Tools,
        )

        tool_calls = response.choices[0].message.tool_calls

        if not tool_calls:
            print("🤖 Assistant:", response.choices[0].message.content)
            break

        log(f"Model requested tool calls : {len(tool_calls)}")

        model_response = response.choices[0].message
        if not model_response.tool_calls :
            print("🤖 Model's Response : ", model_response.content)
            messages.append({
                "role" : "assistant",
                "content" : model_response.content,
            }) 

            break

        # ?Append the model response to the messages 
        # !I dont know why this model_dump is used here
        messages.append(model_response.model_dump(exclude_none=True))

        for tool_call in model_response.tool_calls : 
            # ?Call the tool and get the response
            func_response = run_tool(tool_call)

            # ?Append the tool response to the messages array so that the model can use it in the next turn
            messages.append({
                "role" : "tool",
                "tool_call_id" : tool_call.id,
                "content" : json.dumps(func_response),
            })

        # ?Calling the 2nd API to get the final response from the model after the tool has been called and the tool response has been added to the messages array

        # ?Now as we have added the while loop this is not necessary as the model will keep calling the tools until it has no more tool calls to make and then it will return the final response to the user
        # final_response = client.chat.completions.create(
        #     model = os.getenv("MODEL"),
        #     messages=messages,
        #     tools = Tools,
        # ) 
    
        # # ?Now add the final_response to the messages array so that the model can use it in the next turn
        # messages.append({
        #     "role" : "assistant",
        #     "content" : final_response.choices[0].message.content,
        # })

        # print("Model response: ", final_response.choices[0].message.content)

    # ?If the model does not want to call any tool, just add the model response to the messages array
    # messages.append({
    #     "role" : "assistant",
    #     "content" : model_response.content,
    # })

    # print("Model response : ", model_response.content)

print("conversation ended.")

