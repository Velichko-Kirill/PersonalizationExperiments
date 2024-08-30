from langchain_core.prompts import PromptTemplate

gen_action_prompt: PromptTemplate = PromptTemplate(
    template="""
        You are playing a role of an ordinary person and your task is to decide what 
        are you going to do for the next hour.
        Here is your brief biography: "{bio}"
        Here is a profile with your key characteristics: {profile}

        Right now it's {timestamp} on a {weekday} and you {action_type}

        This is what you were doing before this: "{prev_action}", take this into account and try not to do the same 
        activity for too long or you will get bored, try to do different things

        Here are some relevant activities you did that can help you plan:
        {memories}

        What are your plans for the next hour? Describe them without going into detail, keep just the facts about what 
        you are planning to do, dont mention your name
        Write just the plans and keep in mind that you cant change location while doing them, only when you are in Transit.

    """, input_variables=["bio", "profile", "timestamp", "weekday", "action_type", "prev_action", "memories"]
)

gen_template_prompt: PromptTemplate = PromptTemplate(
    template="""
        You need to generate a person's schedule, which is a text file that contains 7 lines like this one:
            r2t1p9t1s2
            Which, when parsed, is equal to:
            8:00 - 10:00 Recreation
            10:00 - 11:00 Transit
            11:00 - 20:00 Professional
            20:00 - 21:00 Transit
            21:00 - 23:00 Social

        It's very important to mention, that there can only be 4 possible activities:
        p - Professional
        r - Recreation
        s - Social
        t - Transit

        Create a profile for a person with given biography and psychological profile, that are provided below: 
        {profile}
        The profile should contain 7 unparsed lines, and please explain your reasoning behind EACH one, for each also 
        provide a day of the week. 
        In your explanation dont just retell the schedule, focus on the explaining the reasoning and specific planned 
        activities. Take into account, that you cant have more than 16 hours in one unparsed line of schedule and 
        that most people dont work on Sundays or Saturdays (but some do). 
        Also, absolutely all the output lines of your answer that dont contain the schedule should start with a symbol #.

    """, input_variables=["profile"]
)