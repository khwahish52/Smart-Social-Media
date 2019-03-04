Dependencies:
*keras
*pickle
*numpy
*preprocessor  $ pip install tweet-preprocessor  (https://github.com/s/preprocessor)
*nltk
*autocorrect


Steps to use the saved classifier:

Place the final.py in the same directory as your own program and write the following snippet.
             
EXAMPLE SNIPPET :

from final import lstm

j=["hello there how are you hope you are fine and doing good", "hello there how are you hope you are fine and doing good", "JesusChrist was STRAIGHT That's why the faggots killed him. PERIOD SonOfGod; ", "RT @FuckBoiRik: Ok seriously fuck the people who slam on their breaks when the light turns yellow. FLOOR THAT SHIT BITCH"]

for i in j:
    print(lstm(i))
