######################################
####### Author : Erdoğan Aydın #######
######################################

# Model
import torch

# GetResponse
import imaplib
import email

# getTime
from datetime import datetime

# speak
from gtts import gTTS
import playsound
import os

# sendMail
import time
import smtplib
import imghdr
import os
from email.message import EmailMessage

# Run
import torchvision.transforms as transforms
import cv2
import time
from datetime import datetime
import numpy as np
import torchvision


class SecurityProtocol():
    """
    SecurityProtocol works to find stranger user from camera with using pytorch model(self.model).
    """
    def __init__(self):
        self.model = torch.load("SecurityProjectModel.pt")
        self.weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.auto_transforms = self.weights.transforms()
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.model.eval()
        self.speak("Security protocol opened","en")
        self.run()
  
    def speak(self,text,lang):  
        """
        This function is to voice the text
        """
        
        tts = gTTS(text = text,lang = lang)
        filename = "voice.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        os.remove(filename)    
        
    def getResponse(self,email_adress,password):
        
        """
        This function providesto listen email till 5 minutes.
        if response is close pc returns true else false

        """
        
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
            # delete last post
            imap.store(latest_message_id, "+FLAGS", "\\Deleted")
            imap.expunge()
        imap.logout()
        return response
    
    def getTime(self):
        
        """
        This function returns how many minutes have passed today.
        """
        
        now = datetime.now()

        current_time = now.strftime("%H:%M")
        hour = current_time.split(":")[0]
        minute = current_time.split(":")[1]
        return int(hour)*60+int(minute)
    
    def sendMail(self,Sender_Email,Reciever_Email,password):
        
        """
        This function sends mail to admin
        """
        
        newMessage = EmailMessage()                         
        newMessage['Subject'] = "Permission denied" 
        newMessage['From'] = Sender_Email                   
        newMessage['To'] = Reciever_Email                      
        newMessage.add_alternative("""
        <html>
          <head></head>
          <body>
            <p>Permission denied !!!!</p>
            <p>Hi boss, it looks like someone has logged into your computer without your permission. Photo attached.... Would you like to close PC</p>
          </body>
        </html>
        """,subtype='html')
        with open('stranger.jpg', 'rb') as f:
            image_data = f.read()
            image_type = imghdr.what(f.name)
            image_name = f.name
        newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(Sender_Email,password)              
            smtp.send_message(newMessage)
            
    def run(self):
        
        """
        This func starts the program
        """
        
        class_names = ['admin','human']
        last_time = self.getTime()-6
        flag = True
        while True:
            try:
                if(self.getTime()-last_time>5):
                    results = {'admin' : 0 , 'human': 0 }  
                    video_capture = cv2.VideoCapture(0)
                    for i in range(20):
                        _, frame = video_capture.read()
                        frame2 = frame.copy()
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = self.preprocess(frame)
                        frame = frame.unsqueeze(0)

                        with torch.no_grad():
                            output = self.model(frame.to(self.device))
                            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                            preds = probabilities>0.50
                            if any(preds):
                                predicted_class = torch.argmax(probabilities).item()
                                if predicted_class == 0:
                                    results["admin"] += 1
                                else:
                                    results["human"] +=1
                            time.sleep(0.35)
                    video_capture.release()
                    if results["admin"]>10:
                        if flag:
                            self.speak("Hello boss . Welcome home","en")
                            flag = False
                        last_time = self.getTime()+15

                    elif results["human"]>10:
                        cv2.imwrite('stranger.jpg',frame2)
                        Sender_Email = ""
                        Reciever_Email = ""
                        self.sendMail(Sender_Email,Reciever_Email,"password")
                        os.remove("stranger.jpg")
                        last_time = self.getTime()

                else:
                    response = self.getResponse("sendermail","password")
                    if response:
                        self.speak("You are not my admin. I am closing the PC","en")
                        time.sleep(1)
                        os.system("shutdown /s /t 1")

            except:
                self.speak("Something went wrong . I am sorry . I report the problem ","en")
                # Will add information mail
                break
            

            
SecurityProtocol()            

