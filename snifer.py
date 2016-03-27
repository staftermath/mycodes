import requests
import smtplib
import getpass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import datetime
import time


class CanonLensSnifer:
    def __init__(self):
        self.itemlibrary = {
            "http://shop.usa.canon.com/shop/en/catalog/ef-500mm-f-4L-is-ii-usm-refurbished-super-telephoto-lens":"500mm f4L IS II",
            "http://shop.usa.canon.com/shop/en/catalog/ef-16-35mm-f4l-is-usm-refurbished":"16-35mm f4L IS",
            "http://shop.usa.canon.com/shop/en/catalog/ef-400mm-f-4-do-is-ii-usm-refurbished-super-telephoto-lens":"EF 400mm f/4 DO IS II USM Refurbished",
            "http://shop.usa.canon.com/shop/en/catalog/ts-e-17mm-f4l-refurbished":"TS-E 17mm f/4L Refurbished",
            "http://shop.usa.canon.com/shop/en/catalog/ts-e-24mm-f35l-ii-refurbished":"TS-E 24mm f/3.5L II Refurbished",
            "http://shop.usa.canon.com/shop/en/catalog/ef-600mm-f4l-is-ii-usm-refurbished-super-telephoto-lens":"EF 600mm f/4L IS II USM Refurbished Super Telephoto Lens",
            # "http://shop.usa.canon.com/shop/en/catalog/ef-s-24mm-f-28-stm-refurbished":"24mm f2.8 STM",
            # "http://shop.usa.canon.com/shop/en/catalog/ef-24-105mm-f-4l-is-usm-refurbished":"EF 24-105mm f/4L IS USM Refurbished",
            "http://shop.usa.canon.com/shop/en/catalog/extender-14x-lll-refurbished":"Extender 1.4x lll Refurbished"}
        self.inStockIndicator = {}
        for url in self.itemlibrary:
            self.inStockIndicator[self.itemlibrary[url]] = 0
        self.__fromEmail = "gwengww@hotmail.com"
        self.__SMTPServer = "smtp.live.com"
        self.toEmail = "gwengww@gmail.com"
        self.password = None

    def addTracker(self, url, item):
        self.itemlibrary[url] = item
        self.inStockIndicator[item] = 0

    def removeTracker(self, item):
        for url in self.itemlibrary:
            if self.itemlibrary[url] == item:
                del self.itemlibrary[url]
                break
        del self.inStockIndicator[item]

    def addInventory(self, item):
        self.inStockIndicator[item] = 1

    def resetInventory(self, item):
        self.inStockIndicator[item] = 0

    def getPWD(self):
        self.password = getpass.getpass("Please enter your password:")

    def sendEmail(self, url, price):
        now = datetime.datetime.now()
        SUBJECT = "{6}! In Stock Alert: {5} on {0:02}:{1:02} {2:02}-{3:02}-{4}".format(
            now.hour, now.minute, now.month, now.day, now.year, self.itemlibrary[url], price)
        msg = MIMEMultipart('alternative')
        msg['Subject'] = SUBJECT
        msg['From'] = self.__fromEmail
        msg['To'] = self.toEmail
        html = """\
        <html>
          <head></head>
          <body>
            <p>Found Item In Stock<br>
               {}<br>
               Here is the <a href={}>link</a> you wanted.
            </p>
          </body>
        </html>
        """.format(self.itemlibrary[url], url)
        body = MIMEText(html, 'html')
        msg.attach(body)
        username = self.__fromEmail
        password = self.password
        server = smtplib.SMTP(self.__SMTPServer, 587)
        server.ehlo()
        server.starttls()
        server.login(username, password)
        server.sendmail(self.__fromEmail, self.toEmail, msg.as_string())
        server.quit()

    def snifer(self, url):
        r = requests.get(url)
        content = r.text
        instockStart = "href=\"http://schema.org/InStock\" content=\"InStock\">"
        start = content.find(instockStart)
        if start > -1:
            if content[start+len(instockStart):start+len(instockStart)+len("In Stock")].lower() == "in stock":
                return True, content
        else:
            return False, None

    def findprice(self,content):
        location = content.find("<span class=\"price final_price\" itemprop=\"price\">")
        length = len("<span class=\"price final_price\" itemprop=\"price\">")
        end = content[location+length:].find("<")
        return content[location+length:location+length+end].strip()

    def runSnifer(self):
        print "Get Password"
        self.getPWD()
        while True:
            time.sleep(5)
            print "Running..."
            for url in self.itemlibrary:
                findit, content = self.snifer(url)
                if findit:
                    if self.inStockIndicator[self.itemlibrary[url]] == 0:
                        price = self.findprice(content)
                        self.sendEmail(url,price)
                        self.addInventory(self.itemlibrary[url])
                else:
                    self.resetInventory(self.itemlibrary[url])


if __name__ == "__main__":
    sf = CanonLensSnifer()
    sf.runSnifer()
