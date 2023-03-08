import smtplib, json
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

with open("emailPassword.json", "r") as json_read:
	password = json.load(json_read)

def send_attached_email(mail_subject, body, attach_file_path):
	mail_content = f"""
	Caption for audio is: \n\n {body}
	"""
	print(password["email"])
	#The mail addresses and password
	sender_address = 'l39X35f828DPWf9j@gmail.com'
	sender_pass = password["email"]
	receiver_address = "nuo_wen_lei@brown.edu, eric_j_han@brown.edu"
	#Setup the MIME
	message = MIMEMultipart()
	message['From'] = sender_address
	message['To'] = receiver_address
	message['Subject'] = mail_subject
	#The subject line
	#The body and the attachments for the mail
	message.attach(MIMEText(mail_content, 'plain'))
	attach_file_name = 'audio.wav'
	attach_file = open(attach_file_path, 'rb') # Open the file as binary mode
	payload = MIMEBase('audio', 'wav')
	payload.set_payload((attach_file).read())
	encoders.encode_base64(payload) #encode the attachment
	#add payload header with filename
	payload.add_header('Content-Disposition', 'attachment', filename=attach_file_name)
	message.attach(payload)
	#Create SMTP session for sending the mail
	session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
	session.starttls() #enable security
	session.login(sender_address, sender_pass) #login with mail_id and password
	text = message.as_string()
	session.sendmail(sender_address, receiver_address, text)
	session.quit()

def send_email(body):
	mail_content = f"""
	{body}
	"""
	#The mail addresses and password
	sender_address = 'l39X35f828DPWf9j@gmail.com'
	sender_pass = password["email"]
	receiver_address = "nuo_wen_lei@brown.edu"
	#Setup the MIME
	message = MIMEText(mail_content, "plain")
	message['From'] = sender_address
	message['To'] = receiver_address
	message['Subject'] = "CCV Run Status"
	#Create SMTP session for sending the mail
	session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
	session.starttls() #enable security
	session.login(sender_address, sender_pass) #login with mail_id and password
	text = message.as_string()
	session.sendmail(sender_address, receiver_address, text)
	session.quit()
