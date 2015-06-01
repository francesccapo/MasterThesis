import email.utils
import smtplib

gmail_user = "xesc88@gmail.com"
gmail_pwd = "password"
FROM = 'xesc88@gmail.com'
TO = ['xesc88@gmail.com'] #must be a list



def send_email(subject, text):
    # Prepare actual message
    message = """\From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), subject, text)
    try:
        #server = smtplib.SMTP(SERVER) 
        server = smtplib.SMTP("smtp.gmail.com", 587) #or port 465 doesn't seem to work!
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        server.sendmail(FROM, TO, message)
        #server.quit()
        server.close()
        print 'Successfully sent the mail'
    except:
        print "Failed to send mail"

