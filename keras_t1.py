
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

# def w_categorical_crossentropy(y_true, y_pred, weights):
#     nb_cl = len(weights)
#     final_mask = K.zeros_like(y_pred[:, 0])
#     y_pred_max = K.max(y_pred, axis=1)
#     print(K.shape(y_pred))
#     y_pred_max = K.reshape(y_pred_max, K.shape(y_pred))
#     y_pred_max_mat = K.equal(y_pred, y_pred_max)
#     for c_p, c_t in product(range(nb_cl), range(nb_cl)):
#
#         final_mask += (K.cast(weights[c_t, c_p],tf.float32) * K.cast(y_pred_max_mat[:, c_p] ,tf.float32)* K.cast(y_true[:, c_t],tf.float32))
#     return K.categorical_crossentropy(y_pred, y_true) * final_mask
#


time1 = time.time()

# con = sqlite3.connect('soccer.sqlite')
# c = con.cursor()

# raw = [x for x in c.execute('SELECT * from match')]

engine = create_engine('mysql+mysqlconnector://root:databaselogin@soccer.colzfodvmhs1.eu-central-1.rds.amazonaws.com/soccerdb')
engine.connect()
print("connected")

x_data = []
y_data = []

spots = {
'H':[1,0,0],
'D':[0,1,0],
'A':[0,0,1]
}

for match in engine.execute("""
SELECT m.FTR,
p1.flanken,p1.abschluss,p1.kopfballPrazision,p1.kurzpasse,p1.volleys,p1.dribbling,p1.effet,p1.freistossPrazision,p1.langePasse,p1.ballkontrolle,p1.beschleunigung,p1.sprintgeschwindigkeit,p1.beweglichkeit,p1.reaktionen,p1.balance,p1.schusskraft,p1.springkraft,p1.ausdauer,p1.starke,p1.fernschusse,p1.aggressivitat,p1.abfangen,p1.stellungsspiel,p1.ubersicht,p1.elfmeter,p1.ruhe,p1.manndeckung,p1.faireZweikampfe,p1.gratsche,p1.twFlugparaden,p1.twFangsicherheit,p1.twAbschlag,p1.twStellungsspiel,p1.twReflexe,
p2.flanken,p2.abschluss,p4.kopfballPrazision,p4.kurzpasse,p4.volleys,p4.dribbling,p4.effet,p4.freistossPrazision,p4.langePasse,p4.ballkontrolle,p4.beschleunigung,p4.sprintgeschwindigkeit,p4.beweglichkeit,p4.reaktionen,p4.balance,p4.schusskraft,p4.springkraft,p4.ausdauer,p4.starke,p4.fernschusse,p4.aggressivitat,p4.abfangen,p4.stellungsspiel,p4.ubersicht,p4.elfmeter,p4.ruhe,p4.manndeckung,p4.faireZweikampfe,p4.gratsche,p4.twFlugparaden,p4.twFangsicherheit,p4.twAbschlag,p4.twStellungsspiel,p4.twReflexe,
p3.flanken,p3.abschluss,p4.kopfballPrazision,p4.kurzpasse,p4.volleys,p4.dribbling,p4.effet,p4.freistossPrazision,p4.langePasse,p4.ballkontrolle,p4.beschleunigung,p4.sprintgeschwindigkeit,p4.beweglichkeit,p4.reaktionen,p4.balance,p4.schusskraft,p4.springkraft,p4.ausdauer,p4.starke,p4.fernschusse,p4.aggressivitat,p4.abfangen,p4.stellungsspiel,p4.ubersicht,p4.elfmeter,p4.ruhe,p4.manndeckung,p4.faireZweikampfe,p4.gratsche,p4.twFlugparaden,p4.twFangsicherheit,p4.twAbschlag,p4.twStellungsspiel,p4.twReflexe,
p4.flanken,p4.abschluss,p4.kopfballPrazision,p4.kurzpasse,p4.volleys,p4.dribbling,p4.effet,p4.freistossPrazision,p4.langePasse,p4.ballkontrolle,p4.beschleunigung,p4.sprintgeschwindigkeit,p4.beweglichkeit,p4.reaktionen,p4.balance,p4.schusskraft,p4.springkraft,p4.ausdauer,p4.starke,p4.fernschusse,p4.aggressivitat,p4.abfangen,p4.stellungsspiel,p4.ubersicht,p4.elfmeter,p4.ruhe,p4.manndeckung,p4.faireZweikampfe,p4.gratsche,p4.twFlugparaden,p4.twFangsicherheit,p4.twAbschlag,p4.twStellungsspiel,p4.twReflexe,
p5.flanken,p5.abschluss,p5.kopfballPrazision,p5.kurzpasse,p5.volleys,p5.dribbling,p5.effet,p5.freistossPrazision,p5.langePasse,p5.ballkontrolle,p5.beschleunigung,p5.sprintgeschwindigkeit,p5.beweglichkeit,p5.reaktionen,p5.balance,p5.schusskraft,p5.springkraft,p5.ausdauer,p5.starke,p5.fernschusse,p5.aggressivitat,p5.abfangen,p5.stellungsspiel,p5.ubersicht,p5.elfmeter,p5.ruhe,p5.manndeckung,p5.faireZweikampfe,p5.gratsche,p5.twFlugparaden,p5.twFangsicherheit,p5.twAbschlag,p5.twStellungsspiel,p5.twReflexe,
p6.flanken,p6.abschluss,p6.kopfballPrazision,p6.kurzpasse,p6.volleys,p6.dribbling,p6.effet,p6.freistossPrazision,p6.langePasse,p6.ballkontrolle,p6.beschleunigung,p6.sprintgeschwindigkeit,p6.beweglichkeit,p6.reaktionen,p6.balance,p6.schusskraft,p6.springkraft,p6.ausdauer,p6.starke,p6.fernschusse,p6.aggressivitat,p6.abfangen,p6.stellungsspiel,p6.ubersicht,p6.elfmeter,p6.ruhe,p6.manndeckung,p6.faireZweikampfe,p6.gratsche,p6.twFlugparaden,p6.twFangsicherheit,p6.twAbschlag,p6.twStellungsspiel,p6.twReflexe,
p7.flanken,p7.abschluss,p7.kopfballPrazision,p7.kurzpasse,p7.volleys,p7.dribbling,p7.effet,p7.freistossPrazision,p7.langePasse,p7.ballkontrolle,p7.beschleunigung,p7.sprintgeschwindigkeit,p7.beweglichkeit,p7.reaktionen,p7.balance,p7.schusskraft,p7.springkraft,p7.ausdauer,p7.starke,p7.fernschusse,p7.aggressivitat,p7.abfangen,p7.stellungsspiel,p7.ubersicht,p7.elfmeter,p7.ruhe,p7.manndeckung,p7.faireZweikampfe,p7.gratsche,p7.twFlugparaden,p7.twFangsicherheit,p7.twAbschlag,p7.twStellungsspiel,p7.twReflexe,
p8.flanken,p8.abschluss,p8.kopfballPrazision,p8.kurzpasse,p8.volleys,p8.dribbling,p8.effet,p8.freistossPrazision,p8.langePasse,p8.ballkontrolle,p8.beschleunigung,p8.sprintgeschwindigkeit,p8.beweglichkeit,p8.reaktionen,p8.balance,p8.schusskraft,p8.springkraft,p8.ausdauer,p8.starke,p8.fernschusse,p8.aggressivitat,p8.abfangen,p8.stellungsspiel,p8.ubersicht,p8.elfmeter,p8.ruhe,p8.manndeckung,p8.faireZweikampfe,p8.gratsche,p8.twFlugparaden,p8.twFangsicherheit,p8.twAbschlag,p8.twStellungsspiel,p8.twReflexe,
p9.flanken,p9.abschluss,p9.kopfballPrazision,p9.kurzpasse,p9.volleys,p9.dribbling,p9.effet,p9.freistossPrazision,p9.langePasse,p9.ballkontrolle,p9.beschleunigung,p9.sprintgeschwindigkeit,p9.beweglichkeit,p9.reaktionen,p9.balance,p9.schusskraft,p9.springkraft,p9.ausdauer,p9.starke,p9.fernschusse,p9.aggressivitat,p9.abfangen,p9.stellungsspiel,p9.ubersicht,p9.elfmeter,p9.ruhe,p9.manndeckung,p9.faireZweikampfe,p9.gratsche,p9.twFlugparaden,p9.twFangsicherheit,p9.twAbschlag,p9.twStellungsspiel,p9.twReflexe,
p10.flanken,p10.abschluss,p10.kopfballPrazision,p10.kurzpasse,p10.volleys,p10.dribbling,p10.effet,p10.freistossPrazision,p10.langePasse,p10.ballkontrolle,p10.beschleunigung,p10.sprintgeschwindigkeit,p10.beweglichkeit,p10.reaktionen,p10.balance,p10.schusskraft,p10.springkraft,p10.ausdauer,p10.starke,p10.fernschusse,p10.aggressivitat,p10.abfangen,p10.stellungsspiel,p10.ubersicht,p10.elfmeter,p10.ruhe,p10.manndeckung,p10.faireZweikampfe,p10.gratsche,p10.twFlugparaden,p10.twFangsicherheit,p10.twAbschlag,p10.twStellungsspiel,p10.twReflexe,
p11.flanken,p11.abschluss,p11.kopfballPrazision,p11.kurzpasse,p11.volleys,p11.dribbling,p11.effet,p11.freistossPrazision,p11.langePasse,p11.ballkontrolle,p11.beschleunigung,p11.sprintgeschwindigkeit,p11.beweglichkeit,p11.reaktionen,p11.balance,p11.schusskraft,p11.springkraft,p11.ausdauer,p11.starke,p11.fernschusse,p11.aggressivitat,p11.abfangen,p11.stellungsspiel,p11.ubersicht,p11.elfmeter,p11.ruhe,p11.manndeckung,p11.faireZweikampfe,p11.gratsche,p11.twFlugparaden,p11.twFangsicherheit,p11.twAbschlag,p11.twStellungsspiel,p11.twReflexe,
p12.flanken,p12.abschluss,p12.kopfballPrazision,p12.kurzpasse,p12.volleys,p12.dribbling,p12.effet,p12.freistossPrazision,p12.langePasse,p12.ballkontrolle,p12.beschleunigung,p12.sprintgeschwindigkeit,p12.beweglichkeit,p12.reaktionen,p12.balance,p12.schusskraft,p12.springkraft,p12.ausdauer,p12.starke,p12.fernschusse,p12.aggressivitat,p12.abfangen,p12.stellungsspiel,p12.ubersicht,p12.elfmeter,p12.ruhe,p12.manndeckung,p12.faireZweikampfe,p12.gratsche,p12.twFlugparaden,p12.twFangsicherheit,p12.twAbschlag,p12.twStellungsspiel,p12.twReflexe,
p13.flanken,p13.abschluss,p13.kopfballPrazision,p13.kurzpasse,p13.volleys,p13.dribbling,p13.effet,p13.freistossPrazision,p13.langePasse,p13.ballkontrolle,p13.beschleunigung,p13.sprintgeschwindigkeit,p13.beweglichkeit,p13.reaktionen,p13.balance,p13.schusskraft,p13.springkraft,p13.ausdauer,p13.starke,p13.fernschusse,p13.aggressivitat,p13.abfangen,p13.stellungsspiel,p13.ubersicht,p13.elfmeter,p13.ruhe,p13.manndeckung,p13.faireZweikampfe,p13.gratsche,p13.twFlugparaden,p13.twFangsicherheit,p13.twAbschlag,p13.twStellungsspiel,p13.twReflexe,
p14.flanken,p14.abschluss,p14.kopfballPrazision,p14.kurzpasse,p14.volleys,p14.dribbling,p14.effet,p14.freistossPrazision,p14.langePasse,p14.ballkontrolle,p14.beschleunigung,p14.sprintgeschwindigkeit,p14.beweglichkeit,p14.reaktionen,p14.balance,p14.schusskraft,p14.springkraft,p14.ausdauer,p14.starke,p14.fernschusse,p14.aggressivitat,p14.abfangen,p14.stellungsspiel,p14.ubersicht,p14.elfmeter,p14.ruhe,p14.manndeckung,p14.faireZweikampfe,p14.gratsche,p14.twFlugparaden,p14.twFangsicherheit,p14.twAbschlag,p14.twStellungsspiel,p14.twReflexe,
p15.flanken,p15.abschluss,p15.kopfballPrazision,p15.kurzpasse,p15.volleys,p15.dribbling,p15.effet,p15.freistossPrazision,p15.langePasse,p15.ballkontrolle,p15.beschleunigung,p15.sprintgeschwindigkeit,p15.beweglichkeit,p15.reaktionen,p15.balance,p15.schusskraft,p15.springkraft,p15.ausdauer,p15.starke,p15.fernschusse,p15.aggressivitat,p15.abfangen,p15.stellungsspiel,p15.ubersicht,p15.elfmeter,p15.ruhe,p15.manndeckung,p15.faireZweikampfe,p15.gratsche,p15.twFlugparaden,p15.twFangsicherheit,p15.twAbschlag,p15.twStellungsspiel,p15.twReflexe,
p16.flanken,p16.abschluss,p16.kopfballPrazision,p16.kurzpasse,p16.volleys,p16.dribbling,p16.effet,p16.freistossPrazision,p16.langePasse,p16.ballkontrolle,p16.beschleunigung,p16.sprintgeschwindigkeit,p16.beweglichkeit,p16.reaktionen,p16.balance,p16.schusskraft,p16.springkraft,p16.ausdauer,p16.starke,p16.fernschusse,p16.aggressivitat,p16.abfangen,p16.stellungsspiel,p16.ubersicht,p16.elfmeter,p16.ruhe,p16.manndeckung,p16.faireZweikampfe,p16.gratsche,p16.twFlugparaden,p16.twFangsicherheit,p16.twAbschlag,p16.twStellungsspiel,p16.twReflexe,
p17.flanken,p17.abschluss,p17.kopfballPrazision,p17.kurzpasse,p17.volleys,p17.dribbling,p17.effet,p17.freistossPrazision,p17.langePasse,p17.ballkontrolle,p17.beschleunigung,p17.sprintgeschwindigkeit,p17.beweglichkeit,p17.reaktionen,p17.balance,p17.schusskraft,p17.springkraft,p17.ausdauer,p17.starke,p17.fernschusse,p17.aggressivitat,p17.abfangen,p17.stellungsspiel,p17.ubersicht,p17.elfmeter,p17.ruhe,p17.manndeckung,p17.faireZweikampfe,p17.gratsche,p17.twFlugparaden,p17.twFangsicherheit,p17.twAbschlag,p17.twStellungsspiel,p17.twReflexe,
p18.flanken,p18.abschluss,p18.kopfballPrazision,p18.kurzpasse,p18.volleys,p18.dribbling,p18.effet,p18.freistossPrazision,p18.langePasse,p18.ballkontrolle,p18.beschleunigung,p18.sprintgeschwindigkeit,p18.beweglichkeit,p18.reaktionen,p18.balance,p18.schusskraft,p18.springkraft,p18.ausdauer,p18.starke,p18.fernschusse,p18.aggressivitat,p18.abfangen,p18.stellungsspiel,p18.ubersicht,p18.elfmeter,p18.ruhe,p18.manndeckung,p18.faireZweikampfe,p18.gratsche,p18.twFlugparaden,p18.twFangsicherheit,p18.twAbschlag,p18.twStellungsspiel,p18.twReflexe,
p19.flanken,p19.abschluss,p19.kopfballPrazision,p19.kurzpasse,p19.volleys,p19.dribbling,p19.effet,p19.freistossPrazision,p19.langePasse,p19.ballkontrolle,p19.beschleunigung,p19.sprintgeschwindigkeit,p19.beweglichkeit,p19.reaktionen,p19.balance,p19.schusskraft,p19.springkraft,p19.ausdauer,p19.starke,p19.fernschusse,p19.aggressivitat,p19.abfangen,p19.stellungsspiel,p19.ubersicht,p19.elfmeter,p19.ruhe,p19.manndeckung,p19.faireZweikampfe,p19.gratsche,p19.twFlugparaden,p19.twFangsicherheit,p19.twAbschlag,p19.twStellungsspiel,p19.twReflexe,
p20.flanken,p20.abschluss,p20.kopfballPrazision,p20.kurzpasse,p20.volleys,p20.dribbling,p20.effet,p20.freistossPrazision,p20.langePasse,p20.ballkontrolle,p20.beschleunigung,p20.sprintgeschwindigkeit,p20.beweglichkeit,p20.reaktionen,p20.balance,p20.schusskraft,p20.springkraft,p20.ausdauer,p20.starke,p20.fernschusse,p20.aggressivitat,p20.abfangen,p20.stellungsspiel,p20.ubersicht,p20.elfmeter,p20.ruhe,p20.manndeckung,p20.faireZweikampfe,p20.gratsche,p20.twFlugparaden,p20.twFangsicherheit,p20.twAbschlag,p20.twStellungsspiel,p20.twReflexe,
p21.flanken,p21.abschluss,p21.kopfballPrazision,p21.kurzpasse,p21.volleys,p21.dribbling,p21.effet,p21.freistossPrazision,p21.langePasse,p21.ballkontrolle,p21.beschleunigung,p21.sprintgeschwindigkeit,p21.beweglichkeit,p21.reaktionen,p21.balance,p21.schusskraft,p21.springkraft,p21.ausdauer,p21.starke,p21.fernschusse,p21.aggressivitat,p21.abfangen,p21.stellungsspiel,p21.ubersicht,p21.elfmeter,p21.ruhe,p21.manndeckung,p21.faireZweikampfe,p21.gratsche,p21.twFlugparaden,p21.twFangsicherheit,p21.twAbschlag,p21.twStellungsspiel,p21.twReflexe,
p22.flanken,p22.abschluss,p22.kopfballPrazision,p22.kurzpasse,p22.volleys,p22.dribbling,p22.effet,p22.freistossPrazision,p22.langePasse,p22.ballkontrolle,p22.beschleunigung,p22.sprintgeschwindigkeit,p22.beweglichkeit,p22.reaktionen,p22.balance,p22.schusskraft,p22.springkraft,p22.ausdauer,p22.starke,p22.fernschusse,p22.aggressivitat,p22.abfangen,p22.stellungsspiel,p22.ubersicht,p22.elfmeter,p22.ruhe,p22.manndeckung,p22.faireZweikampfe,p22.gratsche,p22.twFlugparaden,p22.twFangsicherheit,p22.twAbschlag,p22.twStellungsspiel,p22.twReflexe,
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
"""):
    for _ in range(5):
        x_data.append([np.random.normal(attr, 3) for attr in match[1:]])
        y_data.append(spots[match[0]])


# for m in raw:
#     home_stats = np.array([x[5:] for x in c.execute('SELECT * FROM team WHERE teamID IS ?', (m[3],) )][0])
#     away_stats = np.array([x[5:] for x in c.execute('SELECT * FROM team WHERE teamID IS ?', (m[4],) )][0])
#
#     m = np.array(m)
#     for _ in range(3):
#         players = []
#         for pid in m[6:17]:
#             if pid == -1 or pid == '-1':
#                 players += [-1]*37
#             else:
#                 players += [a[4:-1] for a in c.execute('SELECT * FROM player WHERE playerID IS ?', (pid,))][0]
#
#         for pid in m[24:35]:
#             if pid == -1:
#                 players += [-1]*37
#             else:
#                 pnew =  [a[4:-1] for a in c.execute('SELECT * FROM player WHERE playerID IS ?', (pid,))]
#                 if len(pnew) == 1:
#                     players += pnew[0]
#                 else:
#                     players += [-1]*37
#
#
#         if np.random.randint(2):
#             players = [np.random.normal(attr, 2) for attr in players]
#
#         x_data.append([*home_stats,*np.array(players).flatten(),*away_stats])
#
#         if m[42] > m[43]:
#             y_data.append([1,0,0])
#         elif m[42] < m[43]:
#             y_data.append([0,0,1])
#         else:
#             y_data.append([0,1,0])
#
#         #shuffle and again
#         np.random.shuffle(m[7:17])
#         np.random.shuffle(m[25:34])


# In[99]:

# for i,t in enumerate(x_data):
#     for j,l in enumerate(t):
#         if l == '':
#             x_data[i][j] = -1
#         elif re.search('^[0-9][0-9]*[+,-][0-9]*',str(l)):
#             k = l.split('-')[0].split('+')[0]
#             x_data[i][j] = k

# corr = [[],[],[]]
# for i,match in enumerate(x_data):
#     fifa_sum = sum(match[9:416])-sum(match[416:-9])
#     if y_data[i][0] == 1:
#         corr[0].append(fifa_sum)
#     elif y_data[i][1] == 1:
#         corr[1].append(fifa_sum)
#     else:
#         corr[2].append(fifa_sum)

x_data = np.array([np.multiply(x,0.01) for x in x_data])

# In[100]:
x_train, x_test , y_train, y_test = train_test_split(x_data, y_data, test_size = 0.20,)
time2 = time.time()
print ('Data Creation took ' + str(int(time2-time1)/60)+'m'+str((time1-time2)%60)+'s')

model = Sequential()
model.add(Dense(units=512, input_dim = len(x_data[0])))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(units=512))
model.add(Activation('relu'))
model.add(Dense(units=512))
# model.add(Activation('sigmoid'))
# model.add(Dense(units=4096))
# model.add(Dense(units=4096))
model.add(Activation('relu'))
# model.add(Dense(units=1024))
# model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(units=3))
model.add(Activation('softmax'))


# In[43]:

len(x_data[0])


# In[61]:
# ncce = partial(w_categorical_crossentropy, weights=w)
# ncce.__name__ ='w_categorical_crossentropy'
model.compile(loss='categorical_crossentropy',
            #   optimizer=RMSprop(lr=0.002, rho=0.9, epsilon=1e-08, decay=0.0),
              optimizer='nadam',
              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=50, batch_size = 1024, class_weight = {k:v for k,v in enumerate(np.mean(y_train, axis=0))},  )
print([int(y) for y in np.sum([x for x in model.predict(x_data[:1000])], axis=0)])
print(np.mean(y_train, axis=0))

for p in [x for x in model.predict(x_data[:100])]:
    print(p)
