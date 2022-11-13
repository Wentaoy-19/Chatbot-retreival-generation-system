import numpy
import openai


def topic_judge(user_utter, topic):
    """
    decide whether the current topic of the user utterance is similar to the current topic
    """
    prompt = """Does the sentence contain the topic? Return True or False. 
        Example 1: 
            Sentence: What is DRAM (DRAM is Dynamic Random Access Memory, so it's a type of memory)
            Topic: Moore machine (Moore machine is a type of machine)
            Return: False (because the memory is different from a machine)
        Example 2:
            Sentence: Where is LA? (LA is Los Angeles, so it's a city)
            Topic: Place (Place is a type of location)
            Return: True (because the city is a type of location)
        Now your turn. 
        Sentence:
        """ + user_utter + "\nTopic: " + topic + "\nReturn:"
    completion = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    text = completion.choices[0].text
    if text.find("True") != -1 or text.find("true") != -1:
        return True
    else:
        return False

def determine_topic(user_utter):
    """
    determine the topic of the current user utterance
    """
    prompt = """
    Determine the topic of the sentence. 
    Example: 
    Sentence: What is Milly Machine? 
    Answer: Milly Machine
    Sentence: Who is Alan Turing? 
    Answer: Alan Turing
    Sentence: 
    """ + user_utter + "\nAnswer:"
    completion = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    text = completion.choices[0].text
    truncated = text.strip()
    return truncated


if __name__ == "__main__":
    current_topic = "turing machine"
    # ask the user to put in sentences
    while True:
        user_utter = input("Please enter a sentence: ")
        if topic_judge(user_utter, current_topic):
            print("The topic is the same.. The current topic is: " + current_topic)
        else:
            current_topic = determine_topic(user_utter)
            print("The topic is different. The new topic is " + current_topic)

# algorithm for high level calling
# ---------- Conversation Starts ----------
# Q: What is Moore machine? (topic: moore machine)
# A: Moore machine is a type of machine
# ---------- Conversation Starts ----------
# Q: What is Turing machine? (topic: turing machine)
# A: Turing machine is a type of machine
# Q: But who designed it? (topic: turing machine)
# A: Alan Turing designed it
# Q: When did he design it? (topic: turing machine)
# A: He designed it in 1936
