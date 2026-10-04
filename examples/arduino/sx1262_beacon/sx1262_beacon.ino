// Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
// SPDX-License-Identifier: MIT
#include <RadioLib.h>

// Semtech SX1262MB2xAS mbed-shield pin map
#define PIN_NSS    D7
#define PIN_DIO1   D5
#define PIN_RESET  A0
#define PIN_BUSY   D3

SX1262 radio = new Module(PIN_NSS, PIN_DIO1, PIN_RESET, PIN_BUSY);

#define BEACON_PERIOD_MS  4000    // one packet every 4 s, measured start to start
const char PAYLOAD[] = "hello world";

#define LRFHSS_CR      RADIOLIB_LR_FHSS_CR_1_3
#define LRFHSS_BW      RADIOLIB_LR_FHSS_BW_39_06
#define LRFHSS_HDRS    3         // header replicas, 1..4 (DR8 uses 3)
#define NARROW_GRID    true      // true = 3.90625 kHz (non-FCC), false = 25.39 kHz
#define HOP_SEQ_ID     0         // legal for every Ngrid

#define TEST_FREQ      915.0     // MHz.
#define TEST_POWER     10        // dBm
#define TCXO_VOLTAGE   0         // 0 = no TCXO (use XTAL).
#define USE_LDO        false

uint32_t beaconCount = 0;
uint32_t nextTxAt    = 0;
uint32_t toaMs       = 0;

void halt(const __FlashStringHelper* what, int code) {
  Serial.print(what);
  Serial.print(F(" FAILED, code "));
  Serial.println(code);
  Serial.println(F("halted"));
  while (true) { delay(1000); }
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && millis() - t0 < 3000) { delay(10); }

  Serial.println();
  Serial.println(F("=== LR-FHSS beacon ==="));

  int state = radio.beginLRFHSS(TEST_FREQ, LRFHSS_BW, LRFHSS_CR, NARROW_GRID,
                                TEST_POWER, TCXO_VOLTAGE, USE_LDO);
  if (state != RADIOLIB_ERR_NONE) halt(F("beginLRFHSS"), state);

  state = radio.setLrFhssConfig(LRFHSS_BW, LRFHSS_CR, LRFHSS_HDRS, HOP_SEQ_ID);
  if (state != RADIOLIB_ERR_NONE) halt(F("setLrFhssConfig"), state);

  toaMs = radio.getTimeOnAir(strlen(PAYLOAD)) / 1000;   // returns microseconds

  Serial.print(F("Payload      : \""));
  Serial.print(PAYLOAD);
  Serial.print(F("\" ("));
  Serial.print(strlen(PAYLOAD));
  Serial.println(F(" bytes)"));
  Serial.print(F("Frequency    : "));
  Serial.print(TEST_FREQ, 3);
  Serial.println(F(" MHz"));
  Serial.println(F("Bandwidth    : 39.06 kHz"));
  Serial.println(F("Coding rate  : 1/3"));
  Serial.print(F("Header reps  : "));
  Serial.println(LRFHSS_HDRS);
  Serial.println(F("Grid         : non-FCC / narrow, 3.90625 kHz step"));
  Serial.print(F("Time on air  : "));
  Serial.print(toaMs);
  Serial.println(F(" ms"));
  Serial.print(F("Period       : "));
  Serial.print(BEACON_PERIOD_MS);
  Serial.println(F(" ms"));

  if (toaMs + 200 > BEACON_PERIOD_MS) {
    Serial.println(F("WARNING: time on air is close to the beacon period"));
  }

  Serial.println();
  nextTxAt = millis();
}

void loop() {
  int32_t wait = (int32_t)(nextTxAt - millis());
  if (wait > 0) { delay(wait); }

  int state = radio.transmit((uint8_t*)PAYLOAD, strlen(PAYLOAD));
  beaconCount++;

  Serial.print(F("BEACON n="));
  Serial.print(beaconCount);
  Serial.print(F(" toa_ms="));
  Serial.print(toaMs);
  Serial.print(F(" code="));
  Serial.println(state);

  nextTxAt += BEACON_PERIOD_MS;
  if ((int32_t)(nextTxAt - millis()) < 0) nextTxAt = millis() + BEACON_PERIOD_MS;
}
