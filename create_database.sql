USE soccerdb;

SET FOREIGN_KEY_CHECKS=0;
DROP TABLE IF EXISTS land, liga, patch, team, player, spiel;
SET FOREIGN_KEY_CHECKS=0;


CREATE TABLE land(
    landID BIGINT PRIMARY KEY,
    name VARCHAR(120)
);

CREATE TABLE liga(
    ligaID BIGINT PRIMARY KEY,
    landID BIGINT,
    name VARCHAR(120),
    FOREIGN KEY(landID) REFERENCES land(landID)
);

CREATE TABLE patch(
    patchID BIGINT PRIMARY KEY,
    sofifaPatchID BIGINT,
    sofifaVersion BIGINT,
    patchDay DATE
);

CREATE TABLE team(
    teamID BIGINT PRIMARY KEY,
    ligaID BIGINT,
    sofifaTeamID BIGINT,
    patchID BIGINT,
    name VARCHAR(120),
    geschwindigkeit BIGINT,
    dribbling BIGINT,
    passen_aufbau BIGINT,
    passen_chancen BIGINT,
    flanken BIGINT,
    schussverhalten BIGINT,
    druck BIGINT,
    aggressivitat BIGINT,
    verschieben BIGINT,
    FOREIGN KEY(ligaID) REFERENCES liga(ligaID),
    FOREIGN KEY(patchID) REFERENCES patch(patchID)
);

CREATE TABLE player(
    playerID BIGINT PRIMARY KEY,
    patchID BIGINT,
    sofifaPlayerID BIGINT,
    teamID BIGINT,
    name VARCHAR(127),
    age BIGINT,
    groesse BIGINT,
    gewicht BIGINT,
    flanken BIGINT,
    abschluss BIGINT,
    kopfballPrazision BIGINT,
    kurzpasse BIGINT,
    volleys BIGINT,
    dribbling BIGINT,
    effet BIGINT,
    freistossPrazision BIGINT,
    langePasse BIGINT,
    ballkontrolle BIGINT,
    beschleunigung BIGINT,
    sprintgeschwindigkeit BIGINT,
    beweglichkeit BIGINT,
    reaktionen BIGINT,
    balance BIGINT,
    schusskraft BIGINT,
    springkraft BIGINT,
    ausdauer BIGINT,
    starke BIGINT,
    fernschusse BIGINT,
    aggressivitat BIGINT,
    abfangen BIGINT,
    stellungsspiel BIGINT,
    ubersicht BIGINT,
    elfmeter BIGINT,
    ruhe BIGINT,
    manndeckung BIGINT,
    faireZweikampfe BIGINT,
    gratsche BIGINT,
    twFlugparaden BIGINT,
    twFangsicherheit BIGINT,
    twAbschlag BIGINT,
    twStellungsspiel BIGINT,
    twReflexe BIGINT,
    jersey BIGINT,
    FOREIGN KEY(teamID) REFERENCES team(teamID),
    FOREIGN KEY(patchID) REFERENCES patch(patchID)
);

CREATE TABLE spiel(
    spielID BIGINT PRIMARY KEY,
    ligaID BIGINT,
    spieltag DATE,
    homeID BIGINT,
    awayID BIGINT,
    url VARCHAR(2500),
    homePlayer1 BIGINT,
    homePlayer2 BIGINT,
    homePlayer3 BIGINT,
    homePlayer4 BIGINT,
    homePlayer5 BIGINT,
    homePlayer6 BIGINT,
    homePlayer7 BIGINT,
    homePlayer8 BIGINT,
    homePlayer9 BIGINT,
    homePlayer10 BIGINT,
    homePlayer11 BIGINT,
    homePlayer12 BIGINT,
    homePlayer13 BIGINT,
    homePlayer14 BIGINT,
    homePlayer15 BIGINT,
    homePlayer16 BIGINT,
    homePlayer17 BIGINT,
    homePlayer18 BIGINT,
    awayPlayer1 BIGINT,
    awayPlayer2 BIGINT,
    awayPlayer3 BIGINT,
    awayPlayer4 BIGINT,
    awayPlayer5 BIGINT,
    awayPlayer6 BIGINT,
    awayPlayer7 BIGINT,
    awayPlayer8 BIGINT,
    awayPlayer9 BIGINT,
    awayPlayer10 BIGINT,
    awayPlayer11 BIGINT,
    awayPlayer12 BIGINT,
    awayPlayer13 BIGINT,
    awayPlayer14 BIGINT,
    awayPlayer15 BIGINT,
    awayPlayer16 BIGINT,
    awayPlayer17 BIGINT,
    awayPlayer18 BIGINT,
    FTHG BIGINT,
    FTAG BIGINT,
    FTR VARCHAR(1),
    HTHG BIGINT,
    HTAG BIGINT,
    HTR VARCHAR(1),
    HShot BIGINT,
    AShot BIGINT,
    HST BIGINT,
    AST BIGINT,
    HF BIGINT,
    AF BIGINT,
    HC BIGINT,
    AC BIGINT,
    HY BIGINT,
    AY BIGINT,
    HR BIGINT,
    AR BIGINT,
    B365H REAL,
    B365D REAL,
    B365A REAL,
    BWH REAL,
    BWD REAL,
    BWA REAL,
    IWH REAL,
    IWD REAL,
    IWA REAL,
    LBH REAL,
    LBD REAL,
    LBA REAL,
    PSH REAL,
    PSD REAL,
    PSA REAL,
    WHH REAL,
    WHD REAL,
    WHA REAL,
    VCH REAL,
    VCD REAL,
    VCA REAL,
    Bb1X2 REAL,
    BbMxH REAL,
    BbAvH REAL,
    BbMxD REAL,
    BbAvD REAL,
    BbMxA REAL,
    BbAvA REAL,
    BbOU REAL,
    BbMxo25 REAL,
    BbAvo25 REAL,
    BbMxu25 REAL,
    BbAvu25 REAL,
    BbAH REAL,
    BbAHh REAL,
    BbMxAHH REAL,
    BbAvAHH REAL,
    BbMxAHA REAL,
    BbAvAHA REAL,
    PSCH REAL,
    PSCD REAL,
    PSCA REAL,
    FOREIGN KEY(ligaID) REFERENCES liga(ligaID),
    FOREIGN KEY(homeID) REFERENCES team(teamID),
    FOREIGN KEY(awayID) REFERENCES team(teamID),
    FOREIGN KEY(homePlayer1) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer2) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer3) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer4) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer5) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer6) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer7) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer8) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer9) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer10) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer11) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer12) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer13) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer14) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer15) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer16) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer17) REFERENCES player(playerID),
    FOREIGN KEY(homePlayer18) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer1) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer2) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer3) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer4) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer5) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer6) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer7) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer8) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer9) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer10) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer11) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer12) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer13) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer14) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer15) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer16) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer17) REFERENCES player(playerID),
    FOREIGN KEY(awayPlayer18) REFERENCES player(playerID)
);


INSERT INTO land VALUES (-1, NULL);
INSERT INTO land VALUES (0, 'International');
INSERT INTO land VALUES (1, 'England');
INSERT INTO land VALUES (2, 'Spanien');
INSERT INTO land VALUES (3, 'Italien');
INSERT INTO land VALUES (4, 'Frankreich');
INSERT INTO land VALUES (5, 'Niederlande');
INSERT INTO land VALUES (6, 'Turkei');
INSERT INTO land VALUES (7, 'Brasilien');
INSERT INTO land VALUES (8, 'USA');
INSERT INTO land VALUES (9, 'Russland');
INSERT INTO land VALUES (10, 'Schweden');
INSERT INTO land VALUES (11, 'Deutschland');

INSERT INTO liga VALUES(-1, -1, '-1');
INSERT INTO liga VALUES(0, 0, 'Nationalmannschaften');
INSERT INTO liga VALUES(1, 0, 'UEFA Champions League');
INSERT INTO liga VALUES(2, 0, 'Europa League');
INSERT INTO liga VALUES(19, 11, 'Bundesliga');
INSERT INTO liga VALUES(20, 11, '2. Bundesliga');
INSERT INTO liga VALUES(13, 1, 'Premier League');
INSERT INTO liga VALUES(14, 1, 'English Championship');
INSERT INTO liga VALUES(53, 2, 'La Liga');
INSERT INTO liga VALUES(31, 3, 'Serie A');
INSERT INTO liga VALUES(16, 4, 'Ligue 1');
INSERT INTO liga VALUES(10, 5, 'Eredevise');
INSERT INTO liga VALUES(68, 6, 'Super Lig');
INSERT INTO liga VALUES(7, 7, 'Brasileiro');
INSERT INTO liga VALUES(39, 8, 'Major League Soccer');
INSERT INTO liga VALUES(67, 9, 'Russische Premier League');
INSERT INTO liga VALUES(56, 10, 'Allsvenskan');
INSERT INTO liga VALUES(60, 1, 'English League One');
INSERT INTO liga VALUES(61, 1, 'English League Two');

INSERT INTO patch VALUES(-1, -1, -1, '2000-01-01');
INSERT INTO patch VALUES(155549, 155549, 9, '2008-08-30');
INSERT INTO patch VALUES(155725, 155725, 9, '2009-02-22');
INSERT INTO patch VALUES(155914, 155914, 10, '2009-08-30');
INSERT INTO patch VALUES(156090, 156090, 10, '2010-02-22');
INSERT INTO patch VALUES(156279, 156279, 11, '2010-08-30');
INSERT INTO patch VALUES(156455, 156455, 11, '2011-02-22');
INSERT INTO patch VALUES(156644, 156644, 12, '2011-08-30');
INSERT INTO patch VALUES(156820, 156820, 12, '2012-02-22');
INSERT INTO patch VALUES(157011, 157011, 13, '2012-08-31');
INSERT INTO patch VALUES(157186, 157186, 13, '2013-02-22');
INSERT INTO patch VALUES(157376, 157376, 14, '2013-08-31');
INSERT INTO patch VALUES(157550, 157550, 14, '2014-02-21');
INSERT INTO patch VALUES(157739, 157739, 15, '2014-08-29');
INSERT INTO patch VALUES(157914, 157914, 15, '2015-02-20');
INSERT INTO patch VALUES(158103, 158103, 16, '2015-08-28');
INSERT INTO patch VALUES(158278, 158278, 16, '2016-02-19');
INSERT INTO patch VALUES(158466, 158466, 17, '2016-08-25');
INSERT INTO patch VALUES(158647, 158647, 17, '2017-02-22');
INSERT INTO patch VALUES(158835, 158835, 18, '2017-08-28');

INSERT INTO team VALUES(-1, -1, -1, -1, '-1', -1, -1, -1, -1, -1, -1, -1, -1, -1);
INSERT INTO player VALUES(-1, -1, -1, -1, '-1', -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
