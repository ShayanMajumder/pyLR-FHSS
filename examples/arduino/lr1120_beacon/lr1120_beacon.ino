// LR-FHSS beacon -- Nucleo-L476RG + LR1120 shield + RadioLib
//
// Board:   Tools > Board > STM32 boards > Nucleo-64 > Nucleo L476RG
// Library: RadioLib (Library Manager)
//
// The same beacon as sx1262_beacon, on an LR1120 instead. Transmits one
// LR-FHSS packet every BEACON_PERIOD_MS, forever, so the receiver has
// something to decode. Keeping the two sketches identical apart from the
// radio is the point: decode both and the difference is the transmitter.
//
// Two things differ from the SX1262 in ways that matter:
//
//  * The LR1120 drives an external RF switch from its own DIO5/DIO6, so a
//    switch table has to be handed to RadioLib. Without it the chip
//    transmits into a switch that is still pointing at the receive path,
//    and almost nothing reaches the antenna -- it looks like a dead
//    radio rather than a configuration mistake.
//  * It has a TCXO. TCXO_VOLTAGE below must match the board, or begin()
//    fails outright.
//
// One line per packet, so a log is machine-readable:
//   BEACON n=<count> toa_ms=<time on air> code=<0 ok, negative RadioLib error>
//
// LR-FHSS is TRANSMIT-ONLY on the LR1120 as well. Decode it with this
// receiver.

#include <RadioLib.h>

// --- wiring ------------------------------------------------------------------
// Semtech LR1120 mbed shield, same header positions as the SX1262 shield.
// If begin() returns -2 (chip not found), this is the first thing to check.
#define PIN_NSS    D7
#define PIN_IRQ    D5      // DIO9
#define PIN_RESET  A0
#define PIN_BUSY   D3

LR1120 radio = new Module(PIN_NSS, PIN_IRQ, PIN_RESET, PIN_BUSY);

// The LR1120 switches its own front end. DIO5/DIO6 is what the Semtech
// shields and most LR11x0 boards use; a different board needs a different
// table, and the symptom of a wrong one is a transmission nobody hears.
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
  { LR11x0::MODE_GNSS,   { LOW,  LOW  } },
  { LR11x0::MODE_WIFI,   { LOW,  LOW  } },
  END_OF_MODE_TABLE,
};

// --- what to transmit --------------------------------------------------------
#define BEACON_PERIOD_MS  4000    // start to start, not end to start
const char PAYLOAD[] = "hello world";

// --- radio configuration -----------------------------------------------------
#define LRFHSS_CR      RADIOLIB_LRXXXX_LR_FHSS_CR_1_3
#define LRFHSS_BW      RADIOLIB_LRXXXX_LR_FHSS_BW_39_06
#define LRFHSS_HDRS    3         // header replicas, 1..4 (DR8 uses 3)
#define NARROW_GRID    true      // true = 3.90625 kHz (non-FCC), false = 25.39 kHz
#define HOP_SEED       0         // the receiver recovers this from the header

// The LR-FHSS sync word, LSB first: RadioLib memcpy()s these four bytes into
// a uint32_t, so this array is 0x12AD101B. THIS IS NOT OPTIONAL. The SX126x
// driver builds its LR-FHSS frame in software with 0x12AD101B hardcoded, but
// the LR11x0 builds the frame in silicon and RadioLib writes the sync word
// only from setSyncWord() -- beginLRFHSS() leaves the chip's register at
// whatever reset put there. Skip this and the radio transmits a structurally
// perfect LR-FHSS frame that no receiver expecting 0x12AD101B can lock onto.
uint8_t SYNC_WORD[4] = { 0x1B, 0x10, 0xAD, 0x12 };

#define TEST_FREQ      915.0     // MHz
#define TEST_POWER     10        // dBm
#define TCXO_VOLTAGE   1.6       // the LR1120 shields have a TCXO; 0 = none

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
  Serial.println(F("=== LR1120 LR-FHSS beacon ==="));

  int state = radio.beginLRFHSS(TEST_FREQ, LRFHSS_BW, LRFHSS_CR, NARROW_GRID,
                                TEST_POWER, TCXO_VOLTAGE);
  if (state != RADIOLIB_ERR_NONE) halt(F("beginLRFHSS"), state);

  // Before anything is transmitted: the table is what points the switch at
  // the antenna. It returns nothing, so a wrong table is silent -- the
  // symptom is a beacon the receiver never hears.
  radio.setRfSwitchTable(rfswitch_dio_pins, rfswitch_table);

  // beginLRFHSS defaults to 3 header replicas; set them explicitly, along
  // with the hop seed, so this sketch says what it transmits.
  state = radio.setLrFhssConfig(LRFHSS_BW, LRFHSS_CR, LRFHSS_HDRS, HOP_SEED);
  if (state != RADIOLIB_ERR_NONE) halt(F("setLrFhssConfig"), state);

  // Must come after beginLRFHSS: setSyncWord dispatches on the active modem,
  // and only routes four bytes to the LR-FHSS sync word once that is LR-FHSS.
  state = radio.setSyncWord(SYNC_WORD, sizeof(SYNC_WORD));
  if (state != RADIOLIB_ERR_NONE) halt(F("setSyncWord"), state);

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
  Serial.print(F("Sync word    : 0x"));
  for (int i = 3; i >= 0; i--) {
    if (SYNC_WORD[i] < 0x10) Serial.print('0');
    Serial.print(SYNC_WORD[i], HEX);
  }
  Serial.println();
  Serial.print(F("Period       : "));
  Serial.print(BEACON_PERIOD_MS);
  Serial.println(F(" ms"));
  Serial.print(F("Reported ToA : "));
  Serial.print(toaMs);
  Serial.println(F(" ms (RadioLib's LR-FHSS estimate is unreliable;"));
  Serial.println(F("               measure from a capture if it matters)"));

  Serial.println();
  nextTxAt = millis();
}

void loop() {
  // Schedule on absolute deadlines rather than delay() after the packet, so
  // the period stays start-to-start instead of drifting by the time on air
  // plus whatever the serial prints cost.
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
  // If we have fallen behind (a failed transmit can block for a while),
  // skip ahead rather than firing a burst of catch-up packets.
  if ((int32_t)(nextTxAt - millis()) < 0) nextTxAt = millis() + BEACON_PERIOD_MS;
}
