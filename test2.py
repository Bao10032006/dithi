from twilio.rest import Client    


client=Client("ACd5b96c466d0bf837c2eadeb8babd0151","5c370971784e306c57ee26ba62ac5241")
message=client.messages.create(
    body="canh bao co nguoi lay do",
    from_ ="+12542674015",
    to="+84363839173"
        )