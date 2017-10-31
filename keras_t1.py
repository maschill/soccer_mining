
# coding: utf-8

# In[20]:
import tensorflow as tf
import pandas as pd
import numpy as np
import sqlite3
import re, time

from sqlalchemy import create_engine

from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
import keras.backend as K

from sklearn.model_selection import train_test_split

from functools import partial
from itertools import product

time1 = time.time()



engine = create_engine('mysql://root:databaselogin@soccer.colzfodvmhs1.eu-central-1.rds.amazonaws.com/soccerdb')
engine.connect()
print("connected")

x_data = []
y_data = []

spots = {
'H':[1,0,0],
'D':[0,1,0],
'A':[0,0,1]
}

matches = [match for match in engine.execute("""
SELECT m.FTR,
p1.flanken,p1.abschluss,p1.kopfballPrazision,p1.kurzpasse,p1.volleys,p1.dribbling,p1.effet,p1.freistossPrazision,p1.langePasse,p1.ballkontrolle,p1.beschleunigung,p1.sprintgeschwindigkeit,p1.beweglichkeit,p1.reaktionen,p1.balance,p1.schusskraft,p1.springkraft,p1.ausdauer,p1.starke,p1.fernschusse,p1.aggressivitat,p1.abfangen,p1.stellungsspiel,p1.ubersicht,p1.elfmeter,p1.ruhe,p1.manndeckung,p1.faireZweikampfe,p1.gratsche,p1.twFlugparaden,
p1.twFangsicherheit,p1.twAbschlag,p1.twStellungsspiel,p1.twReflexe,
p2.flanken,p2.abschluss,p2.kopfballPrazision,p2.kurzpasse,p2.volleys,p2.dribbling,p2.effet,p2.freistossPrazision,p2.langePasse,p2.ballkontrolle,p2.beschleunigung,p2.sprintgeschwindigkeit,p2.beweglichkeit,p2.reaktionen,p2.balance,p2.schusskraft,p2.springkraft,p2.ausdauer,p2.starke,p2.fernschusse,p2.aggressivitat,p2.abfangen,p2.stellungsspiel,p2.ubersicht,p2.elfmeter,p2.ruhe,p2.manndeckung,p2.faireZweikampfe,p2.gratsche,
p3.flanken,p3.abschluss,p3.kopfballPrazision,p3.kurzpasse,p3.volleys,p3.dribbling,p3.effet,p3.freistossPrazision,p3.langePasse,p3.ballkontrolle,p3.beschleunigung,p3.sprintgeschwindigkeit,p3.beweglichkeit,p3.reaktionen,p3.balance,p3.schusskraft,p3.springkraft,p3.ausdauer,p3.starke,p3.fernschusse,p3.aggressivitat,p3.abfangen,p3.stellungsspiel,p3.ubersicht,p3.elfmeter,p3.ruhe,p3.manndeckung,p3.faireZweikampfe,p3.gratsche,
p4.flanken,p4.abschluss,p4.kopfballPrazision,p4.kurzpasse,p4.volleys,p4.dribbling,p4.effet,p4.freistossPrazision,p4.langePasse,p4.ballkontrolle,p4.beschleunigung,p4.sprintgeschwindigkeit,p4.beweglichkeit,p4.reaktionen,p4.balance,p4.schusskraft,p4.springkraft,p4.ausdauer,p4.starke,p4.fernschusse,p4.aggressivitat,p4.abfangen,p4.stellungsspiel,p4.ubersicht,p4.elfmeter,p4.ruhe,p4.manndeckung,p4.faireZweikampfe,p4.gratsche,
p5.flanken,p5.abschluss,p5.kopfballPrazision,p5.kurzpasse,p5.volleys,p5.dribbling,p5.effet,p5.freistossPrazision,p5.langePasse,p5.ballkontrolle,p5.beschleunigung,p5.sprintgeschwindigkeit,p5.beweglichkeit,p5.reaktionen,p5.balance,p5.schusskraft,p5.springkraft,p5.ausdauer,p5.starke,p5.fernschusse,p5.aggressivitat,p5.abfangen,p5.stellungsspiel,p5.ubersicht,p5.elfmeter,p5.ruhe,p5.manndeckung,p5.faireZweikampfe,p5.gratsche,
p6.flanken,p6.abschluss,p6.kopfballPrazision,p6.kurzpasse,p6.volleys,p6.dribbling,p6.effet,p6.freistossPrazision,p6.langePasse,p6.ballkontrolle,p6.beschleunigung,p6.sprintgeschwindigkeit,p6.beweglichkeit,p6.reaktionen,p6.balance,p6.schusskraft,p6.springkraft,p6.ausdauer,p6.starke,p6.fernschusse,p6.aggressivitat,p6.abfangen,p6.stellungsspiel,p6.ubersicht,p6.elfmeter,p6.ruhe,p6.manndeckung,p6.faireZweikampfe,p6.gratsche,
p7.flanken,p7.abschluss,p7.kopfballPrazision,p7.kurzpasse,p7.volleys,p7.dribbling,p7.effet,p7.freistossPrazision,p7.langePasse,p7.ballkontrolle,p7.beschleunigung,p7.sprintgeschwindigkeit,p7.beweglichkeit,p7.reaktionen,p7.balance,p7.schusskraft,p7.springkraft,p7.ausdauer,p7.starke,p7.fernschusse,p7.aggressivitat,p7.abfangen,p7.stellungsspiel,p7.ubersicht,p7.elfmeter,p7.ruhe,p7.manndeckung,p7.faireZweikampfe,p7.gratsche,
p8.flanken,p8.abschluss,p8.kopfballPrazision,p8.kurzpasse,p8.volleys,p8.dribbling,p8.effet,p8.freistossPrazision,p8.langePasse,p8.ballkontrolle,p8.beschleunigung,p8.sprintgeschwindigkeit,p8.beweglichkeit,p8.reaktionen,p8.balance,p8.schusskraft,p8.springkraft,p8.ausdauer,p8.starke,p8.fernschusse,p8.aggressivitat,p8.abfangen,p8.stellungsspiel,p8.ubersicht,p8.elfmeter,p8.ruhe,p8.manndeckung,p8.faireZweikampfe,p8.gratsche,
p9.flanken,p9.abschluss,p9.kopfballPrazision,p9.kurzpasse,p9.volleys,p9.dribbling,p9.effet,p9.freistossPrazision,p9.langePasse,p9.ballkontrolle,p9.beschleunigung,p9.sprintgeschwindigkeit,p9.beweglichkeit,p9.reaktionen,p9.balance,p9.schusskraft,p9.springkraft,p9.ausdauer,p9.starke,p9.fernschusse,p9.aggressivitat,p9.abfangen,p9.stellungsspiel,p9.ubersicht,p9.elfmeter,p9.ruhe,p9.manndeckung,p9.faireZweikampfe,p9.gratsche,
p10.flanken,p10.abschluss,p10.kopfballPrazision,p10.kurzpasse,p10.volleys,p10.dribbling,p10.effet,p10.freistossPrazision,p10.langePasse,p10.ballkontrolle,p10.beschleunigung,p10.sprintgeschwindigkeit,p10.beweglichkeit,p10.reaktionen,p10.balance,p10.schusskraft,p10.springkraft,p10.ausdauer,p10.starke,p10.fernschusse,p10.aggressivitat,p10.abfangen,p10.stellungsspiel,p10.ubersicht,p10.elfmeter,p10.ruhe,p10.manndeckung,p10.faireZweikampfe,p10.gratsche,
p11.flanken,p11.abschluss,p11.kopfballPrazision,p11.kurzpasse,p11.volleys,p11.dribbling,p11.effet,p11.freistossPrazision,p11.langePasse,p11.ballkontrolle,p11.beschleunigung,p11.sprintgeschwindigkeit,p11.beweglichkeit,p11.reaktionen,p11.balance,p11.schusskraft,p11.springkraft,p11.ausdauer,p11.starke,p11.fernschusse,p11.aggressivitat,p11.abfangen,p11.stellungsspiel,p11.ubersicht,p11.elfmeter,p11.ruhe,p11.manndeckung,p11.faireZweikampfe,p11.gratsche,
p12.flanken,p12.abschluss,p12.kopfballPrazision,p12.kurzpasse,p12.volleys,p12.dribbling,p12.effet,p12.freistossPrazision,p12.langePasse,p12.ballkontrolle,p12.beschleunigung,p12.sprintgeschwindigkeit,p12.beweglichkeit,p12.reaktionen,p12.balance,p12.schusskraft,p12.springkraft,p12.ausdauer,p12.starke,p12.fernschusse,p12.aggressivitat,p12.abfangen,p12.stellungsspiel,p12.ubersicht,p12.elfmeter,p12.ruhe,p12.manndeckung,p12.faireZweikampfe,p12.gratsche,p12.twFlugparaden,
p12.twFangsicherheit,p12.twAbschlag,p12.twStellungsspiel,p12.twReflexe,
p13.flanken,p13.abschluss,p13.kopfballPrazision,p13.kurzpasse,p13.volleys,p13.dribbling,p13.effet,p13.freistossPrazision,p13.langePasse,p13.ballkontrolle,p13.beschleunigung,p13.sprintgeschwindigkeit,p13.beweglichkeit,p13.reaktionen,p13.balance,p13.schusskraft,p13.springkraft,p13.ausdauer,p13.starke,p13.fernschusse,p13.aggressivitat,p13.abfangen,p13.stellungsspiel,p13.ubersicht,p13.elfmeter,p13.ruhe,p13.manndeckung,p13.faireZweikampfe,p13.gratsche,
p14.flanken,p14.abschluss,p14.kopfballPrazision,p14.kurzpasse,p14.volleys,p14.dribbling,p14.effet,p14.freistossPrazision,p14.langePasse,p14.ballkontrolle,p14.beschleunigung,p14.sprintgeschwindigkeit,p14.beweglichkeit,p14.reaktionen,p14.balance,p14.schusskraft,p14.springkraft,p14.ausdauer,p14.starke,p14.fernschusse,p14.aggressivitat,p14.abfangen,p14.stellungsspiel,p14.ubersicht,p14.elfmeter,p14.ruhe,p14.manndeckung,p14.faireZweikampfe,p14.gratsche,
p15.flanken,p15.abschluss,p15.kopfballPrazision,p15.kurzpasse,p15.volleys,p15.dribbling,p15.effet,p15.freistossPrazision,p15.langePasse,p15.ballkontrolle,p15.beschleunigung,p15.sprintgeschwindigkeit,p15.beweglichkeit,p15.reaktionen,p15.balance,p15.schusskraft,p15.springkraft,p15.ausdauer,p15.starke,p15.fernschusse,p15.aggressivitat,p15.abfangen,p15.stellungsspiel,p15.ubersicht,p15.elfmeter,p15.ruhe,p15.manndeckung,p15.faireZweikampfe,p15.gratsche,
p16.flanken,p16.abschluss,p16.kopfballPrazision,p16.kurzpasse,p16.volleys,p16.dribbling,p16.effet,p16.freistossPrazision,p16.langePasse,p16.ballkontrolle,p16.beschleunigung,p16.sprintgeschwindigkeit,p16.beweglichkeit,p16.reaktionen,p16.balance,p16.schusskraft,p16.springkraft,p16.ausdauer,p16.starke,p16.fernschusse,p16.aggressivitat,p16.abfangen,p16.stellungsspiel,p16.ubersicht,p16.elfmeter,p16.ruhe,p16.manndeckung,p16.faireZweikampfe,p16.gratsche,
p17.flanken,p17.abschluss,p17.kopfballPrazision,p17.kurzpasse,p17.volleys,p17.dribbling,p17.effet,p17.freistossPrazision,p17.langePasse,p17.ballkontrolle,p17.beschleunigung,p17.sprintgeschwindigkeit,p17.beweglichkeit,p17.reaktionen,p17.balance,p17.schusskraft,p17.springkraft,p17.ausdauer,p17.starke,p17.fernschusse,p17.aggressivitat,p17.abfangen,p17.stellungsspiel,p17.ubersicht,p17.elfmeter,p17.ruhe,p17.manndeckung,p17.faireZweikampfe,p17.gratsche,
p18.flanken,p18.abschluss,p18.kopfballPrazision,p18.kurzpasse,p18.volleys,p18.dribbling,p18.effet,p18.freistossPrazision,p18.langePasse,p18.ballkontrolle,p18.beschleunigung,p18.sprintgeschwindigkeit,p18.beweglichkeit,p18.reaktionen,p18.balance,p18.schusskraft,p18.springkraft,p18.ausdauer,p18.starke,p18.fernschusse,p18.aggressivitat,p18.abfangen,p18.stellungsspiel,p18.ubersicht,p18.elfmeter,p18.ruhe,p18.manndeckung,p18.faireZweikampfe,p18.gratsche,
p19.flanken,p19.abschluss,p19.kopfballPrazision,p19.kurzpasse,p19.volleys,p19.dribbling,p19.effet,p19.freistossPrazision,p19.langePasse,p19.ballkontrolle,p19.beschleunigung,p19.sprintgeschwindigkeit,p19.beweglichkeit,p19.reaktionen,p19.balance,p19.schusskraft,p19.springkraft,p19.ausdauer,p19.starke,p19.fernschusse,p19.aggressivitat,p19.abfangen,p19.stellungsspiel,p19.ubersicht,p19.elfmeter,p19.ruhe,p19.manndeckung,p19.faireZweikampfe,p19.gratsche,
p20.flanken,p20.abschluss,p20.kopfballPrazision,p20.kurzpasse,p20.volleys,p20.dribbling,p20.effet,p20.freistossPrazision,p20.langePasse,p20.ballkontrolle,p20.beschleunigung,p20.sprintgeschwindigkeit,p20.beweglichkeit,p20.reaktionen,p20.balance,p20.schusskraft,p20.springkraft,p20.ausdauer,p20.starke,p20.fernschusse,p20.aggressivitat,p20.abfangen,p20.stellungsspiel,p20.ubersicht,p20.elfmeter,p20.ruhe,p20.manndeckung,p20.faireZweikampfe,p20.gratsche,
p21.flanken,p21.abschluss,p21.kopfballPrazision,p21.kurzpasse,p21.volleys,p21.dribbling,p21.effet,p21.freistossPrazision,p21.langePasse,p21.ballkontrolle,p21.beschleunigung,p21.sprintgeschwindigkeit,p21.beweglichkeit,p21.reaktionen,p21.balance,p21.schusskraft,p21.springkraft,p21.ausdauer,p21.starke,p21.fernschusse,p21.aggressivitat,p21.abfangen,p21.stellungsspiel,p21.ubersicht,p21.elfmeter,p21.ruhe,p21.manndeckung,p21.faireZweikampfe,p21.gratsche,
p22.flanken,p22.abschluss,p22.kopfballPrazision,p22.kurzpasse,p22.volleys,p22.dribbling,p22.effet,p22.freistossPrazision,p22.langePasse,p22.ballkontrolle,p22.beschleunigung,p22.sprintgeschwindigkeit,p22.beweglichkeit,p22.reaktionen,p22.balance,p22.schusskraft,p22.springkraft,p22.ausdauer,p22.starke,p22.fernschusse,p22.aggressivitat,p22.abfangen,p22.stellungsspiel,p22.ubersicht,p22.elfmeter,p22.ruhe,p22.manndeckung,p22.faireZweikampfe,p22.gratsche,
t1.geschwindigkeit, t1.dribbling, t1.passen_aufbau, t1.passen_chancen, t1.flanken, t1.schussverhalten, t1.druck, t1.aggressivitat, t1.verschieben, t2.geschwindigkeit, t2.dribbling, t2.passen_aufbau, t2.passen_chancen, t2.flanken, t2.schussverhalten, t2.druck, t2.aggressivitat, t2.verschieben
FROM spiel m
join player p1 on m.homePlayer1 = p1.playerID
join player p2 on m.homePlayer2 = p2.playerID
join player p3 on m.homePlayer3 = p3.playerID
join player p4 on m.homePlayer4 = p4.playerID
join player p5 on m.homePlayer5 = p5.playerID
join player p6 on m.homePlayer6 = p6.playerID
join player p7 on m.homePlayer7 = p7.playerID
join player p8 on m.homePlayer8 = p8.playerID
join player p9 on m.homePlayer9 = p9.playerID
join player p10 on m.homePlayer10 = p10.playerID
join player p11 on m.homePlayer11 = p11.playerID
join player p12 on m.awayPlayer1 = p12.playerID
join player p13 on m.awayPlayer2 = p13.playerID
join player p14 on m.awayPlayer3 = p14.playerID
join player p15 on m.awayPlayer4 = p15.playerID
join player p16 on m.awayPlayer5 = p16.playerID
join player p17 on m.awayPlayer6 = p17.playerID
join player p18 on m.awayPlayer7 = p18.playerID
join player p19 on m.awayPlayer8 = p19.playerID
join player p20 on m.awayPlayer9 = p20.playerID
join player p21 on m.awayPlayer10 = p21.playerID
join player p22 on m.awayPlayer11 = p22.playerID
join team t1 on m.homeID = t1.teamID
join team t2 on m.awayID = t2.teamID
""")]

match_train, match_val = train_test_split(matches, test_size = 0.05,)

for match in match_train:
    size = len(match)-1
    for _ in range(5):
        x_data.append(np.add(np.random.normal(0, 3, size), match[1:]))
        y_data.append(spots[match[0]])



x_val = []
y_val = []
for match in match_val:
    x_val.append(match[1:])
    y_val.append(spots[match[0]])

x_data = np.array([np.multiply(x,0.01) for x in x_data])

x_train, x_test , y_train, y_test = train_test_split(x_data, y_data, test_size = 0.20,)
time2 = time.time()
print ('Data Creation took ' + str(int((time2-time1)/60))+'m '+str((time1-time2)%60)+'s')

model = Sequential()
model.add(Dense(units=256, input_dim = len(x_data[0])))
model.add(Activation('tanh'))
model.add(Dense(units=256))
model.add(Activation('tanh'))
# model.add(Dense(units=256))
# model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(units=256))
model.add(Activation('tanh'))
# model.add(Dense(units=256))
# model.add(Activation('tanh'))
model.add(Dense(units=3))
model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',
            #   optimizer=RMSprop(lr=0.002, rho=0.9, epsilon=1e-08, decay=0.0),q
              optimizer='nadam',
              metrics=['accuracy', 'kullback_leibler_divergence'])


model.fit(x_train, y_train, epochs=15, batch_size = 1024, class_weight = {k:v for k,v in enumerate(np.mean(y_train, axis=0))},  )
# print([int(y) for y in np.sum([x for x in model.predict(x_data[:1000])], axis=0)])
sol = {0:[1,0,0], 1:[0,1,0], 2:[0,0,1]}
stuff = []
conf_mat = [[0,0,0],[0,0,0],[0,0,0]]
print("\n")
print('Durchschnittswerte im Trainingsset: ')
print(np.mean(y_train, axis=0))
print('Durchschnittswerte im Validationset: ')
print(np.mean(y_val, axis=0))
print('Vorhersagen(Auszug): ')
pred = [0,0,0]
predictions = []
print('--Home---+---Draw---+---Away--')
for i,p in enumerate([x for x in model.predict(x_val)]):
    pred = np.add(pred, np.multiply(y_val[i],sol[np.argmax(p)]))
    stuff.append(np.subtract(p,y_val[i])**2)
    conf_mat[np.argmax(p)] = np.add(conf_mat[np.argmax(p)],y_val[i])
    if i%100 == 0:
        print('%6f | %6f | %6f' % (p[0],p[1],p[2]))
    predictions.append(p)

print('x-y**2: %.3f | %.3f | %.3f' % (np.mean(stuff, axis=0)[0],np.mean(stuff, axis=0)[1],np.mean(stuff, axis=0)[2]))
print('correct: ', pred ,', insgesamt: ',sum(pred),'/',len(x_val),' = ',sum(pred)/len(x_val))
print('+-----+-----+-----+')
for l in conf_mat:
    print('| %3d | %3d | %3d |' %(l[0], l[1], l[2]))
print('+-----+-----+-----+')

from sklearn import metrics

print(metrics.roc_auc_score(y_val, predictions))

model.save('soccer_pred_v01.h5')
