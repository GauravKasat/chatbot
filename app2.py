import aisuite as ai
client = ai.Client()
import os
os.environ['HUGGINGFACE_TOKEN'] = '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'
import json
import regex as re
import ast
class ContextualChatbot:
    def __init__(self):
        self.conversation_history = []
        self.max_history_length = 10
    
    def update_conversation_history(self,role,content):
        self.conversation_history.append({"role":role,"content":content})

        if len(self.conversation_history)>self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        
    def generate_response(self,user_input):
        system_prompt="""
                You are a helpful assistant. You need to give responses based on the flow of conversation provided.
                You will be provided a history of user bot conversation but you would need to focus strictly on the most recent (last) message provided in the conversation.
                While Giving response also consider previous history for knowing information about user such as his/her name and other details respectively.
                
                To give efficient and relevant response Follow the below steps
                1. Determine what phase is in conversation right now based on the recent conversation happened.
                2. After determining the phase follow the flow strictly , give responses similiar to example responses provided in workflow.
                
                NOTE1:
                1. Determining the Phase will be Important and in case of incorrect classifications you will be highly penalized.
                2. After determining the Phase your response should also be from the respective phase detected strictly.
                3. Use the "instruction to follow" for your own understanding , do not output this while generating response.

                ### Validation Points:
                a. Think step by step.
                b. Eliminate personal biases and adhere strictly to sample flow of conversation provided.
                c. Use only the content provided in the messages without assuming any external information.
                d. Detect and consider slang, irony, and sarcasm in messages.
                e. Ensure impartiality in your analysis.
                f. Maintain consistency. 
                g. While generating the response do not output {instruction to follow conten}, only ouput responses similiar to example responses provided in the respective phase.

                #######
                SAMPLE FLOW = 
                '''
                1. INTROUDCTION PHASE
                Instruction to follow -> Welcome the user based on their initial greeting. 
                If you know the person's name already only then respond
                    "Hi! Am I  speaking with Mr./Ms. [Name]?"
                Else Politely ask for there name
                Pause for the user's response. 
                If they ask how you are, acknowledge and respond accordingly.

                2. USER IDENTITY CONFIRMATION PHASE:
                
                If the user confirms their identity, only then proceed to appreciate their previous contributions:
                Instruction to follow -> Pause and acknowledge the user's response.
                Example response ->  "Good [Morning/Afternoon], Mr./Ms. [Name]. This is 'Gaurav' from Ketto. Firstly, I would like to thank you for the previous donations you’ve made. Your contribution has saved lives, and we sincerely appreciate your support."
                
                3. EXPLAINING SIP & PRODUCTS PHASE:

                Transition to explaining the SIP program:
                Instruction to follow -> Ensure to pause and wait for the user's response. If the user hesitates or has questions, provide additional details.
                Example -> "Today, I’ve called regarding an emergency campaign. We’re running a Social Impact Plan (SIP), a monthly auto-debit donation initiative. This supports medical and educational needs for underprivileged children. Contributions can be as small as ₹200-₹300, and they come with an 80G tax benefit."
                
                4. HANDLE OBJECTIONS PHASE:

                If the user hesitates or raises objections, handle them with empathy and provide reassurance:
                Instruction to follow -> Attempt to address the concern up to three times using logical and emotional appeals, ensuring not to pressure the user.
                Example response -> "Is there anything specific stopping you from contributing? Your support makes a real difference."
                
                5. CONFIRMATION OF ENROLLMENT PHASE:

                If the user agrees to enroll, confirm the process:
                Instruction to follow -> Guide the user through the steps and confirm when done.
                Example response -> "That’s great to hear! As you’ve previously used [payment method], I’ll send you a request directly on [UPI/WhatsApp]. Once you accept, the process will be complete. You’ll also receive an RBI confirmation for auto-pay. Please don’t cancel it, as it helps sustain the cause for at least 3-6 months."
                
                6. UPSELL ADDITIONAL CAMPAIGNS PHASE: 

                Instruction to follow ->  If the user enrolls, suggest additional campaigns based on their interest. Adapt the response based on their interest.
                Example response -> "Mr./Ms. [Name], since you’ve enrolled in the medical campaign, we also have initiatives for education, food and hunger, animal welfare, elderly support, and women empowerment. Would you like to hear more about any of these?"
                
                7. COLLECT FEEDBACK AND CLOSE PHASE: 
                
                Before ending the call, gather feedback and additional details for future communication:
                Instruction to follow -> Close the conversation with gratitude: like "Thank you so much, Mr./Ms. [Name]. Have a wonderful day!"
                Example response -> "Thank you for your support! Before we end, may I ask a few quick questions to enhance your donation experience? (e.g., preferred cause, frequency, DOB, PAN for 80G benefits.) Can we contact you in the future for other campaigns?"


                NOTE2:
                1. Do not output the flow statements, only generate response similiar to example responses provided in each phase strictly.
                2. If you dont know the name of user politely ask the user.
                2. Follow the flow strictly else you will be highly penalised for generating other messages other then the one's required.
                '''
        """
        self.update_conversation_history("user", user_input)
        try:
            response = client.chat.completions.create(
            model="huggingface:meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},

                *self.conversation_history
            ]
        )
            assistant_response = response.choices[0].message.content.strip()
            self.update_conversation_history("assistant", assistant_response)
            return assistant_response
        except Exception as e:
            print(f"An error occurred: {e}")
            return "I'm sorry, but I encountered an error. Please try again."
    
    def classify_intent(self,user_input):
        system_prompt = 'You are an expert assistant great at classification based on given guidelines'
        user_prompt = """
        As an expert AI model your task is to analyze and classify the intent for the messages based on the following instructions strictly into one of the ['Introduction','Hesitation','Contribution','Ending'].
        ### Guidelines
        1. If the message expresses Greetings/introduction or contains keywords like hi, hello, how are you classify as 'Introduction' strictly.
        2. If the message expresses hesitation classify as 'Hesitation' strictly.
        3. If the message expresses that user is agreeing to contribute then classify as 'Contribution' strictly.
        4. If the message expresses emotions like ending of a conversation then classify as 'Ending' strictly.
        5. If the message contains words like bye, goodbye then classify as 'Ending' strictly.
        
        ### Validation Points:
        a. Think step by step.
        b. Eliminate personal biases and adhere strictly to the guidelines.
        c. Use only the content provided in the messages without assuming any external information.
        d. Detect and consider slang, irony, and sarcasm in messages.
        e. Ensure impartiality in your analysis.
        f. Maintain consistency and accuracy. You will be penalized for incorrect classifications.
        g. Provide your output in a structured JSON format.

        ### Output Format:
        ```json
        {{
            "Intent Annotation": "<Introduction, Hesitation , Contribution, Ending>",
            "Reason for Annotation": "<Brief analysis>",
        }}
        '''
        Input Message : {0}"""

        dj = user_prompt.format(user_input)
        #print(dj)
        response = client.chat.completions.create(
            model="huggingface:meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role":"user","content":user_prompt.format(*user_input)}
            ]
        )
        _cleaned_response = re.sub('\n|```','',response.choices[0].message.content)
        _cleaned_response = re.sub('json','',_cleaned_response)
        print("response is " + _cleaned_response + '\n')
        try:
            _cleaned_response = re.findall('{.*?}', _cleaned_response)[0]
        except:
            _cleaned_response = re.findall('{.*?}', _cleaned_response)

        try:
            json_obj = json.loads(_cleaned_response)
        except:
            json_string = '{"Intent Annotation": "Invalid", "Reason for Annotation": "Aiveyi man hogya!!!"}'

            # Convert JSON string to Python dictionary
            json_obj = json.loads(json_string)
            #json_obj = json.loads("{'Intent Annotation':'Invalid','Reason for Annotation':'Aiveyi man hogya!!!'}")
        
        
        return json_obj['Intent Annotation']


    def run(self):
        print("Chatbot: Hello! How can I assist you today?")
        while True:
            user_input = input("You: ")
            #intent_classify = self.classify_intent(user_input)
            #print('Intent classified is ' + intent_classify)
            response = self.generate_response(user_input)
            print(f"Chatbot: {response}")
                
    
if __name__ == "__main__":
   chatbot = ContextualChatbot()
   chatbot.run()