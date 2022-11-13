import numpy
import openai


def topic_judge(user_utter, topic):
    """
    decide whether the current topic of the user utterance is similar to the current topic
    """
    prompt = """Does the sentence contain the topic? Return True or False. \n
        Example: \n
            Sentence: Where is LA? \n
            Topic: Los Angeles \n
            Return: True \n
            Sentence: What is Moore machine \n
            Topic: milly machine \n\
            Return: False \n
        Now your turn. Sentence: " 
        """ + user_utter + "\nTopic: " + topic + "\nReturn:"
    completion = openai.Completion.create(engine="davinci", prompt=prompt, temperature=0.9, max_tokens=20, top_p=1, frequency_penalty=0, presence_penalty=0.6, stop=["\n"])
    return completion.choices[0].text


def determine_topic(user_utter):
    """
    determine the topic of the current user utterance
    """
    prompt = "Determine the topic of the sentence. Example: What is Milly Machine? (Milly Machine); Who is Alan Turing? (Alan Turing) \nSentence: " + user_utter
    completion = openai.Completion.create(engine="davinci", prompt=prompt, temperature=0.9, max_tokens=20, top_p=1, frequency_penalty=0, presence_penalty=0.6, stop=["\n"])
    return completion.choices[0].text


if __name__ == "__main__":
    current_topic = "Moore Machine"
    # ask the user to put in sentences
    while True:
        user_utter = input("Please enter a sentence: ")
        if topic_judge(user_utter, current_topic):
            print("The topic is the same.. The current topic is: " + current_topic)
        else:
            current_topic = determine_topic(user_utter)
            print("The topic is different. The new topic is " + current_topic)

