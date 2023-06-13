######################################
####### Author : Erdoğan Aydın #######
######################################



import torch
model = torch.load("SecurityProjectModel.h5")
device = "cuda" if torch.cuda.is_available() else "cpu" 


######################### Functions ######################### 


"""
This function providesto listen email till 5 minutes.
if response is close pc returns true else false

"""

import imaplib
import email
def getResponse():

    email_adress = "email"
    password = "password"

    imap_server = "imap.gmail.com"
    imap_port = 993

    imap = imaplib.IMAP4_SSL(imap_server, imap_port)

    imap.login(email_adress, password)

    imap.select("INBOX")

    response, message_ids = imap.search(None, "ALL")
    latest_message_id = message_ids[0].split()[-1]

    _, msg_data = imap.fetch(latest_message_id, "(RFC822)")
    raw_email = msg_data[0][1]
    email_message = email.message_from_bytes(raw_email)

    email_content = ""

    for part in email_message.walk():
        if part.get_content_type() == "text/plain":
            email_content = part.get_payload(decode=True).decode(part.get_content_charset())
            break

    
    response = str(email_content).strip().lower()[:8]=="close pc"
    if response:
        imap.store(latest_message_id, "+FLAGS", "\\Deleted")
        imap.expunge()
    imap.logout()
    return response



"""
This function returns how many minutes have passed today.
"""

from datetime import datetime
def getTime():
    now = datetime.now()
    
    current_time = now.strftime("%H:%M")
    hour = current_time.split(":")[0]
    minute = current_time.split(":")[1]
    return int(hour)*60+int(minute)



"""
This function is to voice the text
"""
from gtts import gTTS
import playsound
import os
def speak(text):
    tts = gTTS(text = text,lang = "en")
    filename = "voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)



"""
This function sends mail to admin
"""
import time
import smtplib
import imghdr
from email.message import EmailMessage
def sendMail(message,image=True,title="Permission Denied !!!!"):
    
    Sender_Email = "email"
    Reciever_Email = "reciever mail"
    newMessage = EmailMessage()                         
    newMessage['Subject'] = title
    newMessage['From'] = Sender_Email                   
    newMessage['To'] = Reciever_Email                      
    newMessage.add_alternative(f"""
    <html>
      <head></head>
      <body>
        <p>{title}</p>
        <p>{message}</p>
      </body>
    </html>
    """,subtype='html')
    
    if image:
        with open('stranger.jpg', 'rb') as f:
            image_data = f.read()
            image_type = imghdr.what(f.name)
            image_name = f.name
        newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)
        
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(Sender_Email,"password")              
        smtp.send_message(newMessage)

#############################################################

"""
This loop starts the program
"""
import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
auto_transforms = weights.transforms()
model.eval()

preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['admin','human']
last_time = getTime()
frame,flag = 0,True

while True:
    
    if(flag or getTime()-last_time>5):
        
        flag=True
        results = {'admin' : 0 , 'human': 0 }  
        
        try:
            video_capture = cv2.VideoCapture(0)
            for i in range(20):
                _, frame = video_capture.read()
                frame2 = frame.copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = preprocess(frame)
                frame = frame.unsqueeze(0)

                # Modelden tahminleri al
                with torch.no_grad():
                    output = model(frame.to(device))
                    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                    preds = probabilities>0.50
                    if any(preds):
                        predicted_class = torch.argmax(probabilities).item()
                        print(predicted_class)
                        if predicted_class == 0:
                            results["admin"] += 1
                        else:
                            results["human"] +=1
                time.sleep(0.5)

            if results["admin"]>10:
                speak("Hello boss . Welcome home")
                flag = False
                last_time = getTime()+15

            elif results["human"]>10:
                cv2.imwrite('stranger.jpg',frame2)
                #################################
                
                sendMail("Hi boss, it looks like someone has logged into your computer without your permission. Photo attached.... Would you like to close PC")


                ###################################
                os.remove("stranger.jpg")
                flag = False
                last_time = getTime()

        except Exception as e:
            speak("Something went wrong . I am sorry . I report the problem")
            sendMail(f"Error : {e}",image=False,title="Something went wrong")
            break
            
    else:
        response = getResponse()
        if response:
            speak("You are not my admin. I am closing the PC")
            time.sleep(1)
            os.system("shutdown /s /t 1")


    video_capture.release()

