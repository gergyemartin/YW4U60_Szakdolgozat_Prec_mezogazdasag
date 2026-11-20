"""

Budapesti Corvinus Egyetem
Gazdaságinformatikus BSc
Gergye Martin - SZAKDOLGOZAT
Precíziós mezőgazdaság üzemgazdasági megtérülésének vizsgálata szimulációs modellel
A szimulált üzemi adatállomány elemzése



FIGYELEM: az elemzés SZIMULÁLT adatokon készül. Az eredmények a modellre
vonatkoznak, nem egy valós gazdaság mért teljesítményére.

AZ ELEMZÉS MENETE
0. A modellbe épített valódi hatás (csak szimulációban ismert, ellenőrzéshez)
1. Leíró statisztika
2. Kétmintás z-próba — és miért félrevezető
3. Függetlenségvizsgálat: véletlen volt-e az átállás? (tábla szinten)
4. Normalitásvizsgálat
5. A megfigyelési egység: miért nem jó a zóna-év?
6. Regresszió: a hozamhatás becslése
7. Mekkora minta kellene a kimutatáshoz?
8. A fedezetkülönbség becslése
9. Kockázat: a kedvezőtlen kimenetek
10. Megtérülés három forgatókönyvben, érzékenységvizsgálat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.weightstats import ztest

adatok = pd.read_excel("szimulalt_adatok.xlsx")
adatok.info()



print("0. A MODELLBE ÉPÍTETT VALÓDI HATÁS (csak szimulációban ismert)")


"""Valós üzemi adatoknál ez a rész nem létezne: a kezelés valódi hatását
senki sem ismeri. Itt azért ismerjük, mert mi építettük a modellbe.
Az alábbi értékek az 01_adatgeneralas.py paramétereinek PONTOS másolatai.
Kizárólag arra használjuk őket, hogy a becsléseket a valódi értékhez
mérjük; a becslések maguk (6-10. pont) NEM használják őket."""

ORSZAGOS_HOZAM = {
    "őszi búza":          {2019: 5.30, 2020: 5.50, 2021: 6.00, 2022: 4.40},
    "kukorica":           {2019: 8.10, 2020: 8.60, 2021: 6.00, 2022: 3.40},
    "napraforgó":         {2019: 3.00, 2020: 2.80, 2021: 2.70, 2022: 1.80},
    "őszi káposztarepce": {2019: 3.00, 2020: 2.80, 2021: 2.80, 2022: 2.50},
}
FELFUTAS = {0: 0.45, 1: 0.75, 2: 1.00, 3: 1.00}
HATAS_ALAP = 0.028
HATAS_HETEROGENITAS = 0.38
HATAS_STRESSZ = 0.055
DIFFERENCIALAS = 0.55
PREC_INPUT_SZORZO = {"műtrágya": 1.005, "vetőmag": 0.970, "növényvédelem": 0.940}
TERULETALAPU_TAMOGATAS = 85000

STRESSZ = {}
for kult in ORSZAGOS_HOZAM:
    legjobb = max(ORSZAGOS_HOZAM[kult].values())
    STRESSZ[kult] = {ev: round(1 - ORSZAGOS_HOZAM[kult][ev] / legjobb, 3)
                     for ev in ORSZAGOS_HOZAM[kult]}


def felfutas(evek_ota):
    # Az átállás óta eltelt évekhez tartozó felfutási szorzó (kezeletlen: 0)
    if pd.isna(evek_ota):
        return 0.0
    return FELFUTAS[min(int(evek_ota), 3)]


# A valódi hozamhatás minden zóna-évre (kezeletlen sorokban 0)"
adatok["valodi_hatas"] = [
    felfutas(sor.evek_atallas_ota) * (HATAS_ALAP
                                      + HATAS_HETEROGENITAS * sor.heterogenitas
                                      + HATAS_STRESSZ * STRESSZ[sor.kultura][sor.ev])
    for sor in adatok.itertuples()]

# Mi lett volna a fedezet a kezelés nélkül? A kezelt sorokból visszaszámoljuk:
# a terményárbevételt elosztjuk (1 + valódi hatás)-sal, a költségtételekből
# pedig kivesszük a zónaszintű differenciálást és a precíziós szorzókat
dif = 1 + DIFFERENCIALAS * (adatok.zona_index - 1.0)
koltseg_kezeles_nelkul = np.where(
    adatok.precizios == 1,
    adatok.mutragya_Ft_ha / (dif * PREC_INPUT_SZORZO["műtrágya"])
    + adatok.vetomag_Ft_ha / (dif * PREC_INPUT_SZORZO["vetőmag"])
    + adatok.novenyvedelem_Ft_ha / PREC_INPUT_SZORZO["növényvédelem"]
    + adatok.egyeb_Ft_ha,
    adatok.koltseg_Ft_ha)
arbevetel_kezeles_nelkul = ((adatok.arbevetel_Ft_ha - TERULETALAPU_TAMOGATAS)
                            / (1 + adatok.valodi_hatas) + TERULETALAPU_TAMOGATAS)
adatok["fedezet_kezeles_nelkul"] = arbevetel_kezeles_nelkul - koltseg_kezeles_nelkul

# Ell.: a kezeletlen sorokban a két fedezetnek egyeznie kell
kezeletlen_sorok = adatok.precizios == 0
print("Kezeletlen sorokban a visszaszámolt fedezet azonos:", np.allclose(adatok.fedezet_kezeles_nelkul[kezeletlen_sorok], adatok.fedezet_Ft_ha[kezeletlen_sorok]))



print("1. LEÍRÓ STATISZTIKA")


print(f"\nMegfigyelések száma: {len(adatok)} (zóna-év)")
print(f"Táblák: {adatok.tabla.nunique()} | Évek: {adatok.ev.min()}-{adatok.ev.max()}")
print(f"Precíziós megfigyelések aránya: {adatok.precizios.mean():.1%}")

print("\nÁtlagos hozam kultúránként és évenként (t/ha):")
print(adatok.pivot_table(index="kultura", columns="ev", values="hozam_t_ha", aggfunc="mean").round(2))
"A 2022-es aszály a tavaszi vetésűeket sújtotta, az őszi búzát kevésbé."

print("\nA precíziós zóna-évek aránya évenként:")
print(adatok.groupby("ev").precizios.mean().round(3))

print("\nA műtrágyaköltség átlaga évenként (Ft/ha, zóna-év szinten):")
print(adatok.groupby("ev").mutragya_Ft_ha.mean().round(0))

# Csoportonkénti összehasonlítás
seged = adatok.groupby("precizios").agg(
    Elemszam=("hozam_t_ha", "count"),
    AtlagHozam=("hozam_t_ha", np.mean),
    AtlagFedezet=("fedezet_Ft_ha", np.mean))
print("\nNyers összehasonlítás kezelés szerint:")
print(seged.round(0))

# A nyers átlagos fedezetkülönbség bootstrap-intervalluma (zóna-év szinten)
nyers_p = adatok.fedezet_Ft_ha[adatok.precizios == 1].values
nyers_k = adatok.fedezet_Ft_ha[adatok.precizios == 0].values
boot_nyers = stats.bootstrap((nyers_p, nyers_k),
                             lambda a, b: np.mean(a) - np.mean(b),
                             confidence_level=0.95, n_resamples=10000,
                             method="percentile", random_state=22)
print(f"\nNyers átlagos fedezetkülönbség (precíziós - konvencionális): {nyers_p.mean() - nyers_k.mean():+,.0f} Ft/ha")
print(f"bootstrap 95%-os intervallum: [{boot_nyers.confidence_interval.low:+,.0f} ; {boot_nyers.confidence_interval.high:+,.0f}]")



print("2. KÉTMINTÁS Z-PRÓBA A HOZAMRA - ÉS MIÉRT FÉLREVEZETŐ")


"Állítás: a precíziós táblák hozama nagyobb."
"1.) H0 és H1"
"H0: Átlag(precíziós) - Átlag(konvencionális) <= 0"
"H1: Átlag(precíziós) - Átlag(konvencionális) > 0"
"Szurkol: H1"

minta_prec = adatok.hozam_t_ha[adatok.precizios == 1]
minta_konv = adatok.hozam_t_ha[adatok.precizios == 0]

"2+3.) Próbafüggvény és p-érték"
"Előfeltétel: mindkét minta nagy (n > 30) --> z-próba használható"
print(f"\nElemszámok: precíziós n={len(minta_prec)}, konvencionális n={len(minta_konv)}")
print("Nagy minta feltétel:", (len(minta_prec) > 30) & (len(minta_konv) > 30))

probafgv, p_ertek = ztest(minta_prec, minta_konv, value=0, alternative="larger")
print(f"\npróbafgv = {probafgv:.3f}")
print(f"p-érték  = {p_ertek:.4f}")

"4.) Döntés"
if p_ertek > 0.10:
    print("p-érték > 10% --> H0-t nem tudjuk elutasítani")
print("\nEZ AZ EREDMÉNY FÉLREVEZETŐ, ÉS EZT A DOLGOZAT MEGMUTATJA:")
print("a két csoport év- és kultúraösszetétele nem azonos: a lépcsőzetes átállás miatt a precíziós megfigyelések súlypontja a későbbi, részben aszályos évekre esik. A nyers összehasonlítás ezért nem a kezelés hatását méri. A 3., 5. és 6. pont ezt a problémát kezeli.")


print("\n3. FÜGGETLENSÉGVIZSGÁLAT: VÉLETLEN VOLT-E AZ ÁTÁLLÁS?")

# Ha a jobb vagy a változatosabb táblák álltak át, akkor a nyers összehasonlítás
# a tábla minőségét méri, nem a technológia hatását. Ezt teszteljük.

# Az átállásról TÁBLÁNKÉNT döntenek, ezért a próba megfigyelési
# egysége a tábla (24 db). Ha zóna-éveket számolnánk, ugyanaz a döntés
# egy táblán belül 16-24-szer szerepelne, mintha független megfigyelés volna

tabla_szint = adatok.groupby("tabla").agg(
    heterogenitas=("heterogenitas", "first"),
    atallt=("precizios", "max")).reset_index()

# A heterogenitást két kategóriába vonjuk össze a táblák mediánja mentén.
het_median = tabla_szint.heterogenitas.median()
tabla_szint["het_kat"] = np.where(tabla_szint.heterogenitas > het_median,
                                  "heterogén", "homogén")
print(f"\nA táblák heterogenitásának mediánja: {het_median:.4f}")

"1.) H0 és H1"
"H0: a sokaságban az átállás független a tábla heterogenitásától"
"H1: a sokaságban összefüggés van"
"Szurkol: H0 (ha független, akkor nincs kiválasztási torzítás)"

kereszt = pd.crosstab(tabla_szint.het_kat, tabla_szint.atallt)
print("\nKereszttábla, TÁBLA szinten (sorok: heterogenitás, oszlopok: átállt-e):")
print(kereszt)
print("\nAz átállt táblák aránya csoportonként:")
print(tabla_szint.groupby("het_kat").atallt.mean().round(3))

"2+3.) Próbafüggvény és p-érték khi-négyzet próbával"
khi_eredmeny = stats.chi2_contingency(kereszt, correction=False)
print(f"\npróbafgv = {khi_eredmeny[0]:.3f}")
print(f"p-érték  = {khi_eredmeny[1]:.4f}")
print(f"szabadságfok = {khi_eredmeny[2]}")

"+1.) Előfeltétel: minden elvi gyakoriság >= 5"
print("Elvi gyakoriságok:\n", np.round(khi_eredmeny[3], 2))
print("Elvi gyakoriságok >= 5:", (khi_eredmeny[3] >= 5).all())

"4.) Döntés"
if khi_eredmeny[1] < 0.05:
    print("\np-érték < 5% --> H0-t elutasítjuk: az átállás összefügg a heterogenitással.")
else:
    print("\np-érték > 5% --> H0-t nem tudjuk elutasítani: az átállás és a heterogenitás között nem mutatható ki összefüggés.")

# ILLUSZTRÁCIÓ: mi történne, ha ROSSZ egységen, zóna-évekre végeznénk el
# ugyanezt a próbát? (Ugyanannak a táblának a zónái itt külön megfigyelésként szerepelnek.)
# A 6.3 alfejezet ezt a hibát mutatja be
adatok["het_kat"] = np.where(adatok.heterogenitas > adatok.heterogenitas.median(),
                             "heterogén", "homogén")
kereszt_zona = pd.crosstab(adatok.het_kat, adatok.precizios)
khi_zona = stats.chi2_contingency(kereszt_zona, correction=False)
print("\nILLUSZTRÁCIÓ, zóna-év szinten (rossz egység):")
print(kereszt_zona)
print(f"próbafgv = {khi_zona[0]:.3f}, p-érték = {khi_zona[1]:.6f}")
print("--> ugyanaz a próba itt 'szignifikáns' lenne, mert 488 egymástól nem független megfigyelést számol függetlennek.")



print("4. NORMALITÁSVIZSGÁLAT A HOZAMOKRA")


# Azt vizsgáljuk, hogy a hozamok normális eloszlásúak-e. Ez azért érdekes, mert több próba előfeltétele a normalitás.

"1.) H0 és H1"
"H0: a hozamok normális eloszlásúak"
"H1: nem normális eloszlásúak"

"2+3.) Kvintilis-módszer: mik lennének az ötödölő értékek normális eloszlás esetén?"
norm_kvint = stats.norm.ppf([0, 0.2, 0.4, 0.6, 0.8, 1],
                            loc=np.mean(adatok.hozam_t_ha),
                            scale=np.std(adatok.hozam_t_ha, ddof=1))

teny_gyak = np.histogram(adatok.hozam_t_ha, bins=norm_kvint)
elvi_gyak = len(adatok) / 5
print(f"\nElvi gyakoriság minden ötödben: {elvi_gyak:.0f}")
print(f"Tényleges gyakoriságok: {teny_gyak[0]}")
print("Előfeltétel (elvi gyakoriság >= 5):", elvi_gyak >= 5)

khi_norm = stats.chisquare(teny_gyak[0], ddof=2)
print(f"\npróbafgv = {khi_norm[0]:.3f}")
print(f"p-érték  = {khi_norm[1]:.6f}")

"4.) Döntés"
if khi_norm[1] < 0.01:
    print("p-érték < 1% --> a hozamok NEM normális eloszlásúak.")
    print("Ennek oka, hogy négy különböző kultúra hozamai vannak egy adatsorban.")
    print("--> Ezért dolgozunk a következő pontban RELATÍV hozammal.")

adatok.hozam_t_ha.hist()
plt.title("A hozamok eloszlása (szimulált adatok)")
plt.xlabel("hozam (t/ha)")
plt.savefig("abra_1_hozameloszlas.png", dpi=150, bbox_inches="tight")
plt.close()




print("5. A MEGFIGYELÉSI EGYSÉG: MIÉRT NEM JÓ A ZÓNA-ÉV?")


# Egy táblán egy évben 4-6 zóna van. Ezek NEM független megfigyelések:
# ugyanaz az időjárás érte őket, ugyanaz a vetőmag, ugyanaz a gazdálkodó.
# Ha 488 zóna-évet független megfigyelésként kezelünk, akkor a becslés
# pontosságát TÚLBECSÜLJÜK — a standard hiba képletében szereplő 'n' nem 488, hanem inkább a tábla-évek száma.

# MEGOLDÁS: a zónákat tábla-év szintre összevonjuk (átlagoljuk).
# Így minden megfigyelés egy önálló tábla-év, amelyeket közelítőleg függetlennek tekintünk 
# (7.3 alfejezet)

tabla_ev = adatok.groupby(["tabla", "ev"]).agg(
    kultura=("kultura", "first"),
    kultura_ev=("kultura_ev", "first"),
    terulet_ha=("terulet_ha", "first"),
    heterogenitas=("heterogenitas", "first"),
    precizios=("precizios", "first"),
    evek_atallas_ota=("evek_atallas_ota", "first"),
    hozam_t_ha=("hozam_t_ha", np.mean),
    mutragya=("mutragya_Ft_ha", np.mean),
    vetomag=("vetomag_Ft_ha", np.mean),
    novenyvedelem=("novenyvedelem_Ft_ha", np.mean),
    tech_koltseg=("tech_koltseg_Ft_ha", np.mean),
    koltseg=("koltseg_Ft_ha", np.mean),
    fedezet=("fedezet_Ft_ha", np.mean),
    valodi_hatas=("valodi_hatas", "first"),
    fedezet_kezeles_nelkul=("fedezet_kezeles_nelkul", np.mean)).reset_index()

print(f"\nZóna-év megfigyelések: {len(adatok)}")
print(f"Tábla-év megfigyelések: {len(tabla_ev)}")
print(f"Ebből precíziós: {tabla_ev.precizios.sum()}, konvencionális: {(1-tabla_ev.precizios).sum()}")
print("\n--> A további elemzést a tábla-év szintű állományon végezzük.")



print("6. REGRESSZIÓ: A HOZAMHATÁS BECSLÉSE")


# Két problémát kell kezelni.

# ELSŐ: a kultúra és az évjárat hatása.
# A négy kultúra hozamszintje nagyságrendekkel eltér (kukorica 8 t/ha,
# napraforgó 3 t/ha), és a 2022-es aszály sem egyformán érintette őket.
# Ezért minden hozamot elosztunk az adott KULTÚRA-ÉV átlagos hozamával.
# Az így kapott relatív hozam átlaga a kezeletlen tábla-évekre minden
# kultúra-évben 1,0, tehát a kultúra és az évjárat hatása kiesik.

# MÁSODIK: a táblák termőhelyi minősége tartósan eltér. Ha az átálló táblák
# véletlenül jobbak vagy rosszabbak, a nyers összehasonlítás ezt is a
# kezelésnek tulajdonítaná. Ezért minden táblára felveszünk egy dummy
# változót. Így a kezelés hatását azokból a táblákból becsüljük, amelyek
# ÁTÁLLTAK: ugyanazon a táblán vetjük össze az átállás előtti és utáni
# éveket, a soha nem kezelt táblák pedig az évjárati alapvonalat adják.

# FONTOS: a kultúra-év átlagot CSAK a kezeletlen
# táblákból számoljuk. Ha a kezelt táblákat is beszámítanánk, akkor azok
# a saját viszonyítási alapjukat emelnék meg, és a becsült hatás lefelé torzulna.

# Mennyi a modellbe TÉNYLEGESEN beépített átlagos hatás? A 2,8% csak az
# alaphatás; ehhez jön a tábla heterogenitásával és az évjárati stresszel
# arányos rész, és az egészet szorozza a tanulási felfutás. A 0. pontban
# a modell paramétereiből pontosan kiszámoltuk.
kezelt = tabla_ev[tabla_ev.precizios == 1].copy()
kezelt["felfutas"] = kezelt.evek_atallas_ota.map(felfutas)
kezelt["stressz"] = [STRESSZ[sor.kultura][sor.ev] for sor in kezelt.itertuples()]
TENYLEGES_HATAS = kezelt.valodi_hatas.mean()
print(f"\nA modellbe TÉNYLEGESEN beépített átlagos hozamhatás a kezelt tábla-éveken: {TENYLEGES_HATAS*100:.2f}%, ebből, a felfutással súlyozva:")
print(f"  alaphatás:              {(kezelt.felfutas * HATAS_ALAP).mean()*100:.2f}%")
print(f"  heterogenitás-tag:      {(kezelt.felfutas * HATAS_HETEROGENITAS * kezelt.heterogenitas).mean()*100:.2f}%")
print(f"  évjárati stressz-tag:   {(kezelt.felfutas * HATAS_STRESSZ * kezelt.stressz).mean()*100:.2f}%")
print(f"  (a felfutás átlagos szorzója: {kezelt.felfutas.mean():.2f})")
print("--> A 2,8% csak a homogén tábla stresszmentes alapesete, NEM az átlag.")

ke_alapvonal = (tabla_ev[tabla_ev.precizios == 0]
                .groupby("kultura_ev").hozam_t_ha.mean())
tabla_ev["hozam_index"] = tabla_ev.hozam_t_ha / tabla_ev.kultura_ev.map(ke_alapvonal)

print(f"\nA relatív hozam átlaga: {tabla_ev.hozam_index.mean():.4f}")
print(f"A relatív hozam szórása: {np.std(tabla_ev.hozam_index, ddof=1):.4f}")

tabla_dummyk = pd.get_dummies(tabla_ev.tabla, prefix="T",
                              drop_first=True, dtype=int)

Y = tabla_ev.hozam_index
X = pd.concat([tabla_ev[["precizios"]], tabla_dummyk], axis=1)
X = sm.add_constant(X)

modell_hozam = sm.OLS(Y, X).fit()

print(f"\nR-négyzet:            {modell_hozam.rsquared:.4f}")
print(f"Korrigált R-négyzet:  {modell_hozam.rsquared_adj:.4f}")
print(f"Globális F-próba p-értéke: {modell_hozam.f_pvalue:.4f}")

b_prec = modell_hozam.params["precizios"]
p_prec = modell_hozam.pvalues["precizios"]
ci_prec = modell_hozam.conf_int(alpha=0.05).loc["precizios"]

print("\nPARCIÁLIS T-PRÓBA A KEZELÉS EGYÜTTHATÓJÁRA")
"H0: B_precizios = 0 (a kezelés nem szignifikáns)"
"H1: B_precizios != 0 (szignifikáns)"
print(f"becsült hozamhatás: {b_prec*100:+.2f}%")
print(f"95%-os konfidencia-intervallum: [{ci_prec[0]*100:+.2f}% ; {ci_prec[1]*100:+.2f}%]")
print(f"p-érték: {p_prec:.4f}")

if p_prec > 0.10:
    print("\np-érték > 10% --> H0-t nem tudjuk elutasítani.")
    print("A HATÁS EZEN A MINTÁN NEM MUTATHATÓ KI.")
elif p_prec < 0.01:
    print("\np-érték < 1% --> H0-t elutasítjuk, a hatás szignifikáns.")
else:
    print("\n1% <= p-érték <= 10% --> a döntés a szignifikancia-szinttől függ.")

# Ell.: Mennyire pontos a becslés a valódi értékhez képest?
print(f"\nA valódi (beépített) hatás:     {TENYLEGES_HATAS*100:.2f}%")
print(f"A becslés eltérése a valóditól: {(b_prec - TENYLEGES_HATAS)*100:+.2f} százalékpont")
print("A konfidencia-intervallum tartalmazza a valódi értéket:", bool(ci_prec[0] <= TENYLEGES_HATAS <= ci_prec[1]))

# Mit változtat, ha NEM szűrjük ki a táblahatást?
X_szuk = sm.add_constant(tabla_ev[["precizios"]])
modell_szuk = sm.OLS(Y, X_szuk).fit()
print(f"\nTábladummyk nélkül a kezelés együtthatója: {modell_szuk.params['precizios']*100:+.2f}% (p = {modell_szuk.pvalues['precizios']:.4f})")
print(f"R-négyzet tábladummyk nélkül: {modell_szuk.rsquared:.4f}")
print(f"R-négyzet tábladummykkal:     {modell_hozam.rsquared:.4f}")
print("A táblahatás kiszűrése nélkül a becslés torzít.")

# Ell.: Miért nagyobb a tábladummyk nélküli becslés? A tartós
# táblahatást az adatgeneráló szkript a tablak.xlsx-be mentette. Valós adatoknál ez nem volna ismert.
tablak = pd.read_excel("tablak.xlsx")
print("\nA tartós táblahatás átlaga (csak szimulációban ismert):")
print(tablak.groupby(tablak.atallas_eve.notna()).tabla_hatas.mean().rename({True: "átálló táblák", False: "nem átálló táblák"}).round(4))


print("\nKIMUTATHATÓ-E A TANULÁSI GÖRBE?")
# Az 1.3 alfejezet szerint a gazdálkodók nem szívesen fordítanak időt a
# rendszerek megismerésére, ezért a technológia a bevezetés évében nem hozza
# a teljes hasznát. A modell ezt 45/75/100%-os felfutással képezi le.
# Most megvizsgáljuk, hogy ez az adatokból kimutatható-e: a kezelés egyetlen
# dummyja helyett az ÁTÁLLÁS ÓTA ELTELT ÉVEK szerint három dummyt veszünk fel.
# A referencia-kategória a kezeletlen tábla-év.

atallas_ev = tabla_ev[tabla_ev.precizios == 1].groupby("tabla").ev.min()
tabla_ev["ota"] = tabla_ev.ev - tabla_ev.tabla.map(atallas_ev)
tabla_ev["ota_kat"] = np.where(tabla_ev.precizios == 0, "kezeletlen",
                       np.where(tabla_ev.ota == 0, "atallas_0ev",
                       np.where(tabla_ev.ota == 1, "atallas_1ev", "atallas_2ev")))

ota_dummyk = pd.get_dummies(tabla_ev.ota_kat, dtype=int)[
    ["atallas_0ev", "atallas_1ev", "atallas_2ev"]]
X_ota = sm.add_constant(pd.concat([ota_dummyk, tabla_dummyk], axis=1))
modell_ota = sm.OLS(Y, X_ota).fit()

print(f"{'átállás óta':<16}{'becsült hatás':>15}{'p-érték':>12}{'n':>6}{'valódi hatás':>15}{'évek':>16}")
for v in ["atallas_0ev", "atallas_1ev", "atallas_2ev"]:
    resz = tabla_ev[tabla_ev.ota_kat == v]
    n_db = int(ota_dummyk[v].sum())
    evek = ", ".join(str(e) for e in sorted(resz.ev.unique()))
    print(f"{v:<16}{modell_ota.params[v]*100:>14.2f}%{modell_ota.pvalues[v]:>12.4f}{n_db:>6}{resz.valodi_hatas.mean()*100:>14.2f}%{evek:>16}")
print("Ha a hatás az évekkel nő, az a tanulási görbével összhangban van. Az egyes részhatások elemszáma kicsi, és a '2 év' kategória csak 2022-es (aszályos) megfigyelésből áll, ezért a felfutás itt nem választható el az évjárati stressztől.")



print("7. MEKKORA MINTA KELLENE A KIMUTATÁSHOZ?")


# A hibahatár képlete: hibahatár = k * SH.
# Két csoport átlagának különbségénél SH = szórás * gyök(1/n1 + 1/n2),
# ami egyenlő csoportméret (m) esetén: SH = szórás * gyök(2/m).
# Egy hatás akkor mutatható ki, ha a hibahatár kisebb a hatásnál: k * szórás * gyök(2/m) < hatás
# Átrendezve: m = 2 * (k * szórás / hatás)^2       [csoportonkénti elemszám]
# A mintaméretet mindig FELFELÉ kerekítjük: 29,5 tábla-év nem elég, 30 kell.

# SZÓRÁS: A regresszió maradéktagjainak szórása. A Bessel-korrekció
# általánosításaként annyi szabadságfokot vonunk le, ahány paramétert a
# regresszió becsült (konstans + kezelés + 23 tábladummy = 25), vagyis
# n - p = 96 - 25 = 71-gyel osztunk, nem n - 1-gyel.
p_param = len(modell_hozam.params)
szoras = np.std(modell_hozam.resid, ddof=p_param)
k = stats.norm.ppf(1 - 0.05 / 2)
print(f"\nBecsült paraméterek száma: {p_param}")
print(f"A modell maradéktagjának szórása (ddof = {p_param}): {szoras:.4f} \n(ugyanez a statsmodels kimenetéből: {np.sqrt(modell_hozam.mse_resid):.4f}; ddof = 1-gyel számolva {np.std(modell_hozam.resid, ddof=1):.4f} lenne: ez alulbecsül)")
print(f"Megbízhatósági szorzó (95%): k = {k:.3f}")

print("\nKimutatandó hatás   Szükséges tábla-év csoportonként (felfelé kerekítve)")
szukseges = {}
for hatas in [0.028, 0.05, TENYLEGES_HATAS, 0.10]:
    m = 2 * (k * szoras / hatas) ** 2
    szukseges[round(hatas, 4)] = int(np.ceil(m))
    jel = "  <- a modell átlagos hatása" if abs(hatas - TENYLEGES_HATAS) < 1e-9 else ""
    print(f"      {hatas*100:>5.2f}%       {m:>8.2f}  ->  {int(np.ceil(m)):>4d}{jel}")

n_prec = int(tabla_ev.precizios.sum())
n_konv = int((1 - tabla_ev.precizios).sum())
print(f"\nA rendelkezésre álló minta: {n_prec} kezelt és {n_konv} kezeletlen tábla-év.")
print("\nEZ A DOLGOZAT EGYIK LEGFONTOSABB EREDMÉNYE:")
print(f"a modell ALAPHATÁSÁNAK (2,8%) kimutatásához csoportonként legalább {szukseges[0.028]} tábla-év megfigyelés kellene, ami több, mint a rendelkezésre álló {n_prec} kezelt tábla-év. A képlet a hibahatár és a hatás egyenlőségéből indul ki, ezért az eredmény ALSÓ becslés.")
print("\nEbből következik, hogy a szakirodalomban közölt, ISMÉTLÉS NÉLKÜLI egyparcellás kísérletek (köztük a 4. fejezetben újraszámolt kettő) lényegesen kisebb mintán állítanak kimutatott hatást, mint amekkora ehhez statisztikailag szükséges volna.")



print("8. A FEDEZETKÜLÖNBSÉG BECSLÉSE")


# A fedezet szintje évjáratonként nagyságrendekkel eltér: 2022-ben az árak
# és a műtrágyaköltség is kiugrott. Mivel a precíziós megfigyelések éppen a
# későbbi évekre esnek, a nyers összehasonlítás itt is összetételi hatást mér.
# A hozamnál osztottunk; a fedezetnél KIVONJUK a kultúra-év átlagát, mert a
# fedezet forintban értelmes és negatív is lehet.

# Itt is a kezeletlen táblák adják az alapvonalat, ugyanazon okból.
ke_fed_alapvonal = (tabla_ev[tabla_ev.precizios == 0]
                    .groupby("kultura_ev").fedezet.mean())
tabla_ev["fedezet_elteres"] = (tabla_ev.fedezet
                               - tabla_ev.kultura_ev.map(ke_fed_alapvonal))

Y_fed = tabla_ev.fedezet_elteres
X_fed = sm.add_constant(pd.concat([tabla_ev[["precizios"]], tabla_dummyk], axis=1))

modell_fedezet = sm.OLS(Y_fed, X_fed).fit()

b_fed = modell_fedezet.params["precizios"]
p_fed = modell_fedezet.pvalues["precizios"]
ci_fed = modell_fedezet.conf_int(alpha=0.05).loc["precizios"]

print(f"\nR-négyzet: {modell_fedezet.rsquared:.4f}")
print("\nA KEZELÉS HATÁSA A FEDEZETRE")
print(f"becsült fedezetkülönbség: {b_fed:+,.0f} Ft/ha")
print(f"95%-os konfidencia-intervallum: [{ci_fed[0]:+,.0f} ; {ci_fed[1]:+,.0f}] Ft/ha")
print(f"p-érték: {p_fed:.4f}")
# Értelmezés: ugyanazon a táblán, a kultúra-év átlagához mérve a precíziós ezelés ennyivel változtatja meg a fedezetet hektáronként.

# A költségoldal: a kezelés hatása tételenként, ugyanezzel a módszerrel"
print("\nA kezelés hatása az egyes költségtételekre (Ft/ha):")
koltseghatas = {}
for tetel in ["mutragya", "vetomag", "novenyvedelem", "tech_koltseg"]:
    alap = (tabla_ev[tabla_ev.precizios == 0].groupby("kultura_ev")[tetel].mean())
    y = tabla_ev[tetel] - tabla_ev.kultura_ev.map(alap)
    m = sm.OLS(y, X_fed).fit()
    koltseghatas[tetel] = m.params["precizios"]
    print(f"  {tetel:<16} {m.params['precizios']:>+9,.0f}   (p = {m.pvalues['precizios']:.3f})")
netto_koltseg = sum(koltseghatas.values())
print(f"  {'nettó':<16} {netto_koltseg:>+9,.0f}")
print("  --> pozitív érték = költségnövekedés = fedezetcsökkenés")
print(f"\nA fedezetkülönbség felbontása: árbevétel-hatás {b_fed + netto_koltseg:+,.0f} ({(b_fed + netto_koltseg) / b_fed:.0%}), költséghatás {-netto_koltseg:+,.0f} ({-netto_koltseg / b_fed:.0%})")

# Ell.: A modellbe épített valódi fedezethatás a kezelt tábla-éveken
TENYLEGES_FEDEZETHATAS = (kezelt.fedezet - kezelt.fedezet_kezeles_nelkul).mean()
print(f"\nA valódi (beépített) fedezethatás: {TENYLEGES_FEDEZETHATAS:+,.0f} Ft/ha")
print(f"A becslés eltérése a valóditól:     {b_fed - TENYLEGES_FEDEZETHATAS:+,.0f} Ft/ha")
print("A konfidencia-intervallum tartalmazza a valódi értéket:", bool(ci_fed[0] <= TENYLEGES_FEDEZETHATAS <= ci_fed[1]))



print("9. KOCKÁZAT: A KEDVEZŐTLEN KIMENETEK")


# A fedezetszámítás az átlagot hasonlítja össze, a gazdálkodó döntése
# viszont a kimenetek egész eloszlására irányul. Itt is ugyanaz a csapda
# áll fenn, mint az 1. és 2. pontban: a precíziós tábla-évek a későbbi,
# drága és aszályos évekre esnek. Ezért először megnézzük a nyers képet,
# utána a kultúra-év alapvonalhoz mért eltéréseket hasonlítjuk össze.

print("\nA fedezet NYERS eloszlása kezelés szerint (Ft/ha, tábla-év szinten):")
print(tabla_ev.groupby("precizios").fedezet.describe(percentiles=[0.05, 0.5, 0.95])[["count", "mean", "5%", "50%", "95%"]].round(0))

# A veszteséges tábla-évek aránya: ez egy ARÁNY, tehát 0/1 kódolás
tabla_ev["veszteseges"] = np.where(tabla_ev.fedezet < 0, 1, 0)

seged_v = tabla_ev.groupby("precizios").agg(
    Arany=("veszteseges", np.mean),
    SH=("veszteseges", stats.sem),
    n=("veszteseges", "count"))
k = stats.norm.ppf(1 - 0.05 / 2)
seged_v["AH"] = seged_v.Arany - seged_v.SH * k
seged_v["FH"] = seged_v.Arany + seged_v.SH * k
print("\nA veszteséges tábla-évek NYERS aránya 95%-os konfidencia-intervallummal:")
print(seged_v.round(4))

felt = ((seged_v.n * seged_v.Arany >= 10) &
        (seged_v.n * (1 - seged_v.Arany) >= 10))
print("\nNagy minta feltétel (n*p >= 10 és n*(1-p) >= 10):")
print(felt.to_string())
if not felt.all():
    print("--> Ahol nem teljesül, ott az arány intervallumbecslése nem megalapozott.")

# Hol vannak a veszteséges tábla-évek? Összetételi hatás-e a különbség?
vesztes_ke = tabla_ev[tabla_ev.veszteseges == 1].kultura_ev.unique()
print(f"\nA veszteséges tábla-évek kultúra-évei: {list(vesztes_ke)}")
resz = tabla_ev[tabla_ev.kultura_ev.isin(vesztes_ke)]
print("Ezekben a kultúra-években (sorok: precíziós-e, oszlopok: veszteséges-e):")
print(pd.crosstab(resz.precizios, resz.veszteseges))
print("Átlagos fedezet ezekben a kultúra-években (Ft/ha):")
print(resz.groupby("precizios").fedezet.agg(["count", "mean"]).round(0))
print("A precíziós tábla-évek között ilyen kultúra-év aránya:", round(tabla_ev[tabla_ev.precizios == 1].kultura_ev.isin(vesztes_ke).mean(), 3), "| a konvencionálisak között:", round(tabla_ev[tabla_ev.precizios == 0].kultura_ev.isin(vesztes_ke).mean(), 3))

# KIIGAZÍTOTT ÖSSZEVETÉS: a fedezet eltérése a kultúra-év kezeletlen
# átlagától (a 8. pontban kiszámolt fedezet_elteres). Így minden tábla-évet
# a saját kultúrájának saját évéhez mérünk, az évösszetétel kiesik.
print("\nA fedezet KIIGAZÍTOTT eloszlása (eltérés a kultúra-év kezeletlen átlagától, Ft/ha):")
print(tabla_ev.groupby("precizios").fedezet_elteres.describe(percentiles=[0.05, 0.5, 0.95])[["count", "mean", "5%", "50%", "95%"]].round(0))

# Bootstrap: a két csoport 5. percentilisének különbsége (kiigazított eloszlás)
# A bootstrap akkor hasznos, ha a mutatóra nincs standard hiba képlet.
# Az 5. percentilis ilyen: azt mutatja, mi történik egy rossz kimenetnél.
fed_p = tabla_ev.fedezet_elteres[tabla_ev.precizios == 1].values
fed_k = tabla_ev.fedezet_elteres[tabla_ev.precizios == 0].values
boot = stats.bootstrap((fed_p, fed_k),
                       lambda a, b: np.quantile(a, 0.05) - np.quantile(b, 0.05),
                       confidence_level=0.95, n_resamples=10000,
                       method="percentile", random_state=22)
print(f"\nAz 5. percentilis különbsége (precíziós - konvencionális, kiigazítva):")
print(f"  pontbecslés: {np.quantile(fed_p,0.05)-np.quantile(fed_k,0.05):+,.0f} Ft/ha")
print(f"  bootstrap 95%-os intervallum: [{boot.confidence_interval.low:+,.0f} ; {boot.confidence_interval.high:+,.0f}]")
print("--> Ha az intervallum tartalmazza a nullát, a rossz kimenetek közötti különbség nem igazolható.")

# Ugyanez az átlagra, hogy lássuk: a rossz kimenet javulása nagyobb-e az átlagénál
boot_atlag = stats.bootstrap((fed_p, fed_k), lambda a, b: np.mean(a) - np.mean(b),
                             confidence_level=0.95, n_resamples=10000,
                             method="percentile", random_state=22)
print(f"\nAz átlag különbsége (kiigazítva): {fed_p.mean() - fed_k.mean():+,.0f} Ft/ha, bootstrap 95%: [{boot_atlag.confidence_interval.low:+,.0f} ; {boot_atlag.confidence_interval.high:+,.0f}]")

plt.figure(figsize=(8, 4.5))
plt.hist(tabla_ev.fedezet_elteres[tabla_ev.precizios == 0] / 1000, bins=15,
         alpha=0.6, label="konvencionális")
plt.hist(tabla_ev.fedezet_elteres[tabla_ev.precizios == 1] / 1000, bins=15,
         alpha=0.6, label="precíziós")
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("eltérés a kultúra-év kezeletlen átlagától (ezer Ft/ha)")
plt.ylabel("gyakoriság (tábla-év)")
plt.title("A fedezet eloszlása a két technológia alatt\n"
          "(szimulált adatok, a kultúra-év kezeletlen átlagához mérve)")
plt.legend()
plt.savefig("abra_3_fedezeteloszlas.png", dpi=150, bbox_inches="tight")
plt.close()



print("10. MEGTÉRÜLÉS HÁROM FORGATÓKÖNYVBEN")


BERUHAZAS = 19_000_000
TAMOGATAS = 0.50
SAJAT_FORRAS = BERUHAZAS * (1 - TAMOGATAS)
ELETTARTAM = 8
MARADVANYERTEK = BERUHAZAS * 0.15
DISZKONTRATA = 0.08

prec_tablak = tabla_ev[tabla_ev.precizios == 1].groupby("tabla").terulet_ha.first()
PREC_TERULET = prec_tablak.sum()
print(f"\nPrecíziós technológiával érintett terület: {PREC_TERULET:.1f} ha")
print(f"Saját forrásigény: {SAJAT_FORRAS:,.0f} Ft")
print(f"Fajlagos beruházás: {SAJAT_FORRAS/PREC_TERULET:,.0f} Ft/ha")


def npv(fedezetkulonbseg, terulet=PREC_TERULET, r=DISZKONTRATA,
        sajat=SAJAT_FORRAS, elettartam=ELETTARTAM, maradvany=MARADVANYERTEK):
    # Nettó jelenérték: a 0. évben kiadás a saját forrás, utána évente
    # a fedezettöbblet, az utolsó évben a maradványérték is.
    # Az élettartam egész évek száma (az éves pénzáramok miatt).
    ertek = -sajat
    for t in range(1, int(elettartam) + 1):
        penz = fedezetkulonbseg * terulet
        if t == int(elettartam):
            penz += maradvany
        ertek += penz / (1 + r) ** t
    return ertek


def irr(fedezetkulonbseg, terulet=PREC_TERULET):
    # Belső megtérülési ráta: az a kamatláb, ahol az NPV nullára vált.
    # 0,1%-os lépésekkel haladunk felfelé (legfeljebb 500%-ig), és az utolsó
    # olyan kamatlábat adjuk vissza, ahol az NPV még nem negatív."""
    if npv(fedezetkulonbseg, terulet, 0.0) < 0:
        return np.nan # már 0%-on sem térül meg
    for r in np.arange(0.001, 5.001, 0.001):
        if npv(fedezetkulonbseg, terulet, r) < 0:
            return r - 0.001
    return np.inf # 500% felett is pozitív


def fedezeti_uzemmeret(fedezetkulonbseg):
    # Az a legkisebb egész hektárszám, ahol az NPV eléri a nullát.
    if fedezetkulonbseg <= 0:
        return np.nan
    for ha in range(1, 20001):
        if npv(fedezetkulonbseg, ha) >= 0:
            return ha
    return np.nan


forgatokonyvek = pd.DataFrame({
    "fedezetkulonbseg": [ci_fed[0], b_fed, ci_fed[1], TENYLEGES_FEDEZETHATAS]},
    index=["CI alsó határa", "pontbecslés", "CI felső határa", "valódi (ellenőrzés)"])
forgatokonyvek["eves_tobblet"] = forgatokonyvek.fedezetkulonbseg * PREC_TERULET
forgatokonyvek["NPV"] = [npv(f) for f in forgatokonyvek.fedezetkulonbseg]
forgatokonyvek["IRR_%"] = [irr(f) * 100 for f in forgatokonyvek.fedezetkulonbseg]
forgatokonyvek["fedezeti_ha"] = [fedezeti_uzemmeret(f)
                                 for f in forgatokonyvek.fedezetkulonbseg]


print(forgatokonyvek.round({"fedezetkulonbseg": 0, "eves_tobblet": 0, "NPV": 0, "IRR_%": 1}).to_string())
print("A GAZDASÁGI KÖVETKEZTETÉS ÖRÖKLI A STATISZTIKAI BECSLÉS BIZONYTALANSÁGÁT: a pontbecslés helyett a konfidencia-intervallum teljes sávját kell közölni. Az utolsó sor mutatja, hogy a valódi érték ezen a mintán a sáv alsó részére esik. Egyetlen pont-NPV ezt elfedné.")

# ----------------------------------------------------------------- ÁBRÁK
meretek = np.arange(25, 2001, 25)
plt.figure(figsize=(8, 4.5))
for nev, sor in forgatokonyvek.iterrows():
    stilus = ":" if nev.startswith("valódi") else "-"
    plt.plot(meretek, [npv(sor.fedezetkulonbseg, m) / 1e6 for m in meretek],
             stilus, label=nev)
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(PREC_TERULET, color="grey", linestyle="--", linewidth=0.8,
            label=f"modellezett üzem ({PREC_TERULET:.0f} ha)")
plt.xscale("log")
plt.xticks([25, 50, 100, 200, 500, 1000, 2000],
           ["25", "50", "100", "200", "500", "1000", "2000"])
plt.xlabel("precíziós technológiával művelt terület (ha, logaritmikus skála)")
plt.ylabel("NPV (millió Ft)")
plt.title("A beruházás nettó jelenértéke az üzemméret függvényében\n(szimulált adatok)")
plt.legend()
plt.savefig("abra_2_fedezeti_uzemmeret.png", dpi=150, bbox_inches="tight")
plt.close()

print("EGYVÁLTOZÓS ÉRZÉKENYSÉGVIZSGÁLAT (+-30%)")
# Minden paramétert egyenként elmozdítunk +-30%-kal, a többit változatlanul
# hagyva, és megnézzük, mennyit mozdul az NPV. Az élettartam egész év, ezért
# ott a legközelebbi egész évet vesszük: 8 x 0,7 = 5,6 -> 6 év, 8 x 1,3 = 10,4 -> 10 év (vagyis ott +-25%)."""

alap_npv = npv(b_fed)
erzekenyseg = []
for nev, kulcs, alapertek in [
        ("Saját forrásigény (Ft)", "sajat", SAJAT_FORRAS),
        ("Precíziós terület (ha)", "terulet", PREC_TERULET),
        ("Fedezetkülönbség (Ft/ha)", "fk", b_fed),
        ("Élettartam (év)", "elettartam", ELETTARTAM),
        ("Diszkontráta", "r", DISZKONTRATA)]:
    ertekek = []
    for szorzo in (0.7, 1.3):
        if kulcs == "fk":
            ertekek.append(npv(b_fed * szorzo))
        elif kulcs == "elettartam":
            ertekek.append(npv(b_fed, elettartam=round(alapertek * szorzo)))
        else:
            ertekek.append(npv(b_fed, **{kulcs: alapertek * szorzo}))
    erzekenyseg.append({"paraméter": nev, "−30%": ertekek[0], "+30%": ertekek[1],
                        "ingadozás": abs(ertekek[1] - ertekek[0])})

erzekenyseg = pd.DataFrame(erzekenyseg).sort_values("ingadozás", ascending=False)
print(erzekenyseg.set_index("paraméter").round(0).to_string())
print(f"\nAlapeset NPV: {alap_npv:,.0f} Ft")
print(f"A legérzékenyebb tényező: {erzekenyseg.iloc[0]['paraméter']}")

# ÉRZÉKENYSÉGVIZSGÁLAT: a fedezeti üzemméret a feltételezett hozamhatás
# függvényében. Ez a legfontosabb ábra, mert a hozamhatás az a bemenet,
# amiben a legnagyobb a bizonytalanság - a dolgozat nem egyetlen értéket
# közöl, hanem megmutatja, hogyan függ a következtetés ettől a bemenettől.

# A hozamhatás csak a TERMÉNYÁRBEVÉTELT növeli (hozam x ár), a hektáronként
# fix területalapú támogatást nem. Ezért a hatást a támogatás nélküli
# árbevételre vetítjük. A költségoldali hatás a 8. pont becsléséből jön.
atl_termeny_arbevetel = (adatok.arbevetel_Ft_ha - TERULETALAPU_TAMOGATAS).mean()
print(f"\nÁtlagos terményárbevétel (támogatás nélkül, 2019-2022): {atl_termeny_arbevetel:,.0f} Ft/ha")
print(f"A precíziós technológia nettó költségnövekedése (8. pont): {netto_koltseg:,.0f} Ft/ha")

hatas_sav = np.arange(0.0025, 0.1051, 0.0025)
meretek_sav = []
for h in hatas_sav:
    fk = atl_termeny_arbevetel * h - netto_koltseg
    meretek_sav.append(fedezeti_uzemmeret(fk) if fk > 0 else np.nan)

plt.figure(figsize=(8, 4.5))
plt.plot(hatas_sav * 100, meretek_sav)
plt.axhline(29, color="green", linestyle=":", label="hazai átlagos szántóterület (29 ha)")
plt.axvline(b_prec * 100, color="red", linestyle="--",
            label=f"a modellből becsült hatás ({b_prec*100:.1f}%)".replace(".", ","))
plt.axvline(TENYLEGES_HATAS * 100, color="grey", linestyle="-.",
            label=f"a modellbe épített valódi hatás ({TENYLEGES_HATAS*100:.1f}%)".replace(".", ","))
plt.ylim(0, 1200)
plt.xlabel("feltételezett hozamhatás (%)")
plt.ylabel("fedezeti üzemméret (ha)")
plt.title("A fedezeti üzemméret a feltételezett hozamhatás függvényében\n(szimulált adatok)")
plt.legend()
plt.savefig("abra_4_erzekenyseg.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nÉRZÉKENYSÉGVIZSGÁLAT - fedezeti üzemméret a hozamhatás függvényében:")
for h in [0.01, 0.02, 0.028, 0.05, TENYLEGES_HATAS, 0.07, b_prec, 0.10]:
    fk = atl_termeny_arbevetel * h - netto_koltseg
    m = fedezeti_uzemmeret(fk) if fk > 0 else None
    if m:
        print(f"  {h*100:>6.2f}%  ->  fedezetkülönbség {fk:>+9,.0f} Ft/ha,  fedezeti üzemméret {m} ha ({m/29:.1f}-szerese a 29 ha-os átlagnak)")
    else:
        print(f"  {h*100:>6.2f}%  ->  fedezetkülönbség {fk:>+9,.0f} Ft/ha,  nem térül meg")
print(f"\n  Az a hozamhatás, amely alatt a technológia semekkora üzemen nem térül meg: {netto_koltseg/atl_termeny_arbevetel*100:.2f}%  (itt a fedezetkülönbség éppen nulla)")
print("\nÁbrák mentve: abra_1, abra_2, abra_3, abra_4 (.png)")

tabla_ev.to_excel("tabla_ev_szint.xlsx", index=False)

# ÖSSZEFOGLALÁS

print("\nÖSSZEFOGLALÁS")
print("\n1. A nyers kétmintás összehasonlítás szerint nem mutatható ki, hogy a precíziós táblák hozama nagyobb (p = {p_ertek:.3f}) — de ez félrevezető: a precíziós megfigyelések a drágább, aszályos évekre esnek. Az átállás tábla szinten nem függ kimutathatóan a heterogenitástól (p = {khi_eredmeny[1]:.3f}).")
print("\n2. A tábla- és évjárathatást kiszűrő regresszió szerint a becsült hozamhatás {b_prec*100:+.2f}%, 95%-os intervallum [{ci_prec[0]*100:+.2f}% ; {ci_prec[1]*100:+.2f}%], p = {p_prec:.3f}. A modellbe épített valódi hatás {TENYLEGES_HATAS*100:.2f}%.")
print("\n3. A 2,8%-os alaphatás kimutatásához csoportonként legalább {szukseges[0.028]} tábla-év kellene, a rendelkezésre álló {n_prec} kezelt tábla-év helyett.")
print("\n4. A becsült fedezetkülönbség {b_fed:+,.0f} Ft/ha (95%: {ci_fed[0]:+,.0f} … {ci_fed[1]:+,.0f} Ft/ha); a valódi {TENYLEGES_FEDEZETHATAS:+,.0f} Ft/ha.")
print("\n5. A pontbecslés mellett az NPV {forgatokonyvek.NPV['pontbecslés']:,.0f} Ft, a fedezeti üzemméret {forgatokonyvek.fedezeti_ha['pontbecslés']} ha — miközben a hazai átlagos szántóterület 29 ha, a medián 5,0 ha.")
print("\nMINDEN EREDMÉNY SZIMULÁLT ADATOKRA VONATKOZIK.")
