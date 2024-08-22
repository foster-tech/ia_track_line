from mailjet_rest import Client

# importing os module for environment variables
import os, time, cv2, base64
# importing necessary functions from dotenv library
from dotenv import load_dotenv, dotenv_values 

class EmailNotifier:
    def __init__(self):
         # Mailjet API credentials
        self.api_key = os.getenv("MAILJET_API_KEY")
        self.api_secret = os.getenv("MAILJET_API_SECRET_KEY")
        self.from_email = os.getenv("MAILJET_FROM_EMAIL")
        self.from_name = os.getenv("MAILJET_FROM_NAME")
        self.interval_seconds = int(os.getenv("MAILJET_INTERVAL")) + 3 # Removing average time to send email 
        self.last_sent_time = 0   

    def send_email(self, attachment_path, to_email, to_name, subject, text):
        # loading variables from .env file
        load_dotenv() 
        current_time = time.time()

        # print(current_time - self.last_sent_time)
        # print(self.interval_seconds)

        # # Check if the interval has passed since the last notification
        # if current_time - self.last_sent_time >= self.interval_seconds:

        with open(attachment_path, "rb") as attachment_file:
            attachment_base64 = base64.b64encode(attachment_file.read()).decode('utf-8')

        # Initialize Mailjet client
        mailjet = Client(auth=(self.api_key, self.api_secret), version='v3.1')

        # Email details
        data = {
            'Messages': [
                {
                    "From": {
                        "Email": self.from_email,
                        "Name": self.from_name
                    },
                    "To": [
                        {
                            "Email": to_email,
                            "Name": to_name
                        }
                    ],
                    # "Cc": [
                    #     {
                    #         "Email": "eduardo.ventura@intelicitybr.com.br",
                    #         "Name": "Eduardo"
                    #     }
                    # ],
                    "Subject": subject,
                    "TextPart": text,
                    "HTMLPart": f'<h3>{text}</h3>',
                    "Attachments": [
                        {
                            "ContentType": "image/png",
                            "Filename": os.path.basename(attachment_path),
                            "Base64Content": attachment_base64
                        }
                    ]
                }
            ]
        }

        # Send email
        # result = mailjet.send.create(data=data)

        # Update the last sent time
        self.last_sent_time = current_time

        # # Print response
        # print(result.status_code)
        # print(result.json())
        # # return True

        # else:
        #     # Check the remaining time and print in the log 
        #     remaining_time = self.interval_seconds - (current_time - self.last_sent_time)
        #     print(f"Notification skipped. Try again in {int(remaining_time)} seconds.")
        #     return False