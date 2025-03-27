import os

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def send_email(email_subject, email_message):
    message = Mail(
        from_email='burakagridag2@gmail.com',
        to_emails='burakagridag@gmail.com',
        subject=email_subject,
        html_content=email_message)
    try:
        SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

        if SENDGRID_API_KEY is None:
            raise ValueError("Missing SendGrid API Key!")

        print("SendGrid API Key loaded successfully!")

        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e.message)

#sendEmail("buy", "hoppaaa")