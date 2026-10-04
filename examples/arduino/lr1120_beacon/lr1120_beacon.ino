// Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
// SPDX-License-Identifier: MIT
#include <RadioLib.h>

// LR1120 mbed shield on the Arduino headers of a Nucleo-L476RG (any Nucleo-64)
#define PIN_NSS    D7
#define PIN_IRQ    D5      // DIO9
#define PIN_RESET  A0
#define PIN_BUSY   D3

LR1120 radio = new Module(PIN_NSS, PIN_IRQ, PIN_RESET, PIN_BUSY);

static const uint32_t rfswitch_dio_pins[] = {
  RADIOLIB_LR11X0_DIO5, RADIOLIB_LR11X0_DIO6,
  RADIOLIB_NC, RADIOLIB_NC, RADIOLIB_NC
};

static const Module::RfSwitchMode_t rfswitch_table[] = {
  // mode                  DIO5  DIO6
  { LR11x0::MODE_STBY,   { LOW,  LOW  } },
  { LR11x0::MODE_RX,     { HIGH, LOW  } },
  { LR11x0::MODE_TX,     { HIGH, HIGH } },
  { LR11x0::MODE_TX_HP,  { LOW,  HIGH } },
  { LR11x0::MODE_TX_HF,  { LOW,  LOW  } },
  { LR11x0::MODE_WIFI,   { LOW,  LOW  } },
  END_OF_MODE_TABLE,
};

const char PAYLOAD[] = "Hello World";

// LoRaWAN EU868 DR8: 136.72 kHz, CR 1/3, 3 header replicas, 3.9 kHz grid
#define LRFHSS_BW      RADIOLIB_LRXXXX_LR_FHSS_BW_136_72
#define LRFHSS_CR      RADIOLIB_LRXXXX_LR_FHSS_CR_1_3
#define LRFHSS_HDRS    3
#define NARROW_GRID    true
#define HOP_SEED       0         // the receiver recovers this from the header

uint8_t SYNC_WORD[4] = { 0x1B, 0x10, 0xAD, 0x12 };   // 0x12AD101B

// 869.4-869.65 MHz allows 500 mW ERP at <= 10% duty cycle. A 1.4 s packet
// every 5 s is ~28%: fine into a dummy load or a cable, not on an antenna.
#define TEST_FREQ        869.525   // MHz
#define TEST_POWER       22        // dBm, the LR1120's maximum (high-power PA)
#define BEACON_PERIOD_MS 5000      // start to start
#define TCXO_VOLTAGE     1.6       // the LR1120 shields have a TCXO; 0 = none

uint32_t beaconCount = 0;
uint32_t nextTxAt    = 0;

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
  Serial.println(F("=== LR1120 LR-FHSS DR8 beacon ==="));

  radio.setRfSwitchTable(rfswitch_dio_pins, rfswitch_table);

  int state = radio.beginLRFHSS(TEST_FREQ, LRFHSS_BW, LRFHSS_CR, NARROW_GRID,
                                TEST_POWER, TCXO_VOLTAGE);
  if (state != RADIOLIB_ERR_NONE) halt(F("beginLRFHSS"), state);
  state = radio.setLrFhssConfig(LRFHSS_BW, LRFHSS_CR, LRFHSS_HDRS, HOP_SEED);
  if (state != RADIOLIB_ERR_NONE) halt(F("setLrFhssConfig"), state);
  state = radio.setSyncWord(SYNC_WORD, sizeof(SYNC_WORD));
  if (state != RADIOLIB_ERR_NONE) halt(F("setSyncWord"), state);
  state = radio.setOutputPower(TEST_POWER);
  if (state != RADIOLIB_ERR_NONE) halt(F("setOutputPower"), state);

  Serial.print(F("Payload      : \""));
  Serial.print(PAYLOAD);
  Serial.println(F("\""));
  Serial.print(F("Frequency    : "));
  Serial.print(TEST_FREQ, 3);
  Serial.println(F(" MHz, DR8 (136.72 kHz, CR 1/3, 3 header replicas)"));
  Serial.print(F("Power        : "));
  Serial.print(TEST_POWER);
  Serial.println(F(" dBm"));
  Serial.print(F("Sync word    : 0x"));
  for (int i = 3; i >= 0; i--) {
    if (SYNC_WORD[i] < 0x10) Serial.print('0');
    Serial.print(SYNC_WORD[i], HEX);
  }
  Serial.println();
  Serial.print(F("Period       : "));
  Serial.print(BEACON_PERIOD_MS);
  Serial.println(F(" ms"));
  Serial.println();
  nextTxAt = millis();
}

void loop() {
  int32_t wait = (int32_t)(nextTxAt - millis());
  if (wait > 0) { delay(wait); }

  int state = radio.transmit((uint8_t*)PAYLOAD, strlen(PAYLOAD));
  Serial.print(F("BEACON n="));
  Serial.print(++beaconCount);
  Serial.print(F(" code="));
  Serial.println(state);

  nextTxAt += BEACON_PERIOD_MS;
  if ((int32_t)(nextTxAt - millis()) < 0) nextTxAt = millis() + BEACON_PERIOD_MS;
}
