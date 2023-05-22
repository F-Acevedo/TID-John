import openai

#Initialize the OpenAI API
openai.api_key = "APIKEY"


def openaiAPI(word_1,word_2):
    response=openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a helpful assistant."},

            {"role": "user", "content": "I need you to make an hypothesis between the word {} and the word {}".format(word_1,word_2)}
        ]
    )
    print(response['choices'][0]['message']['content'])














