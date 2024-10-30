import smtplib
from email.mime.text import MIMEText
msg = MIMEText('Programme on Alienware has finished running.')
msg['Subject'] = 'Alienware done'
msg['From'] = 'chiuyang.python@gmail.com'
msg['To'] = 'chiuyang.python@gmail.com'
with smtplib.SMTP_SSL('smtp.gmail.com',465) as smtp_server:
	smtp_server.login('chiuyang.python@gmail.com', 'zlrgnrrfatjgcrbk')
	smtp_server.sendmail('chiuyang.python@gmail.com','chiuyang.python@gmail.com',\
		msg.as_string())