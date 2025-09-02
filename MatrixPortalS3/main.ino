// === Pin Definitions for MatrixPortal ESP32-S3 ===
const uint8_t PIN_R1 = 42;  // Red top
const uint8_t PIN_G1 = 41;
const uint8_t PIN_B1 = 40;
const uint8_t PIN_R2 = 38;
const uint8_t PIN_G2 = 39;
const uint8_t PIN_B2 = 37;

const uint8_t PIN_A = 45;
const uint8_t PIN_B = 36;
const uint8_t PIN_C = 48;
const uint8_t PIN_D = 35;
const uint8_t PIN_E = 21;

const uint8_t PIN_CLK   = 2;
const uint8_t PIN_LAT   = 47;
const uint8_t PIN_OE    = 14;

// Speed control variable
uint8_t pixelRunnerSpeed = 16;

// Frame counter for completed sweeps
uint16_t frameCounter = 0;

// Button pin definitions for Matrix Portal S3
const uint8_t PIN_BTN_UP   = 6;  // board.BUTTON_UP, Arduino pin 6
const uint8_t PIN_BTN_DOWN = 7;  // board.BUTTON_DOWN, Arduino pin 7

// Speed delay range: lower delay = faster movement
const uint16_t SPEED_DELAY_MIN = 10;   // fastest
const uint16_t SPEED_DELAY_MAX = 100;  // slowest

// LED on-time durations for brightness control - controlled by speed
uint16_t WHITE_ON_TIME = 20;  // µs ON time for movable white LED (variable based on speed)
const uint16_t BAR_ON_TIME   = 20;   // µs ON time for speed/frame bars

// Speed ranges for button control
const uint16_t SPEED_MIN_TIME = 10;   // fastest (10µs)
const uint16_t SPEED_MAX_TIME = 200;  // slowest (200µs)

// === Panel Constants ===
const uint8_t WIDTH  = 32;
const uint8_t HEIGHT = 32;
const uint8_t SCAN_LINES = 16; // 1/16 scan
const uint8_t ROW_PAIR_COUNT = HEIGHT / SCAN_LINES;  // 2 rows per address

uint8_t row = 0;
uint8_t col = 0;

void setup() {
  // Setup RGB, Address, Control Pins
  pinMode(PIN_R1, OUTPUT); pinMode(PIN_G1, OUTPUT); pinMode(PIN_B1, OUTPUT);
  pinMode(PIN_R2, OUTPUT); pinMode(PIN_G2, OUTPUT); pinMode(PIN_B2, OUTPUT);

  pinMode(PIN_A, OUTPUT); pinMode(PIN_B, OUTPUT); pinMode(PIN_C, OUTPUT);
  pinMode(PIN_D, OUTPUT); pinMode(PIN_E, OUTPUT);

  pinMode(PIN_CLK, OUTPUT);
  pinMode(PIN_LAT, OUTPUT);
  pinMode(PIN_OE, OUTPUT);

  // Configure speed control buttons
  pinMode(PIN_BTN_UP, INPUT_PULLUP);
  pinMode(PIN_BTN_DOWN, INPUT_PULLUP);

  digitalWrite(PIN_OE, HIGH);  // Start with output disabled
  digitalWrite(PIN_LAT, LOW);
  digitalWrite(PIN_CLK, LOW);
}

void setRowAddress(uint8_t row) {
  digitalWrite(PIN_A, (row >> 0) & 0x01);
  digitalWrite(PIN_B, (row >> 1) & 0x01);
  digitalWrite(PIN_C, (row >> 2) & 0x01);
  digitalWrite(PIN_D, (row >> 3) & 0x01);
  digitalWrite(PIN_E, (row >> 4) & 0x01); // Not used on 1/16 scan
}

void pulseClock() {
  digitalWrite(PIN_CLK, HIGH);
  digitalWrite(PIN_CLK, LOW);
}

void pulseLatch() {
  digitalWrite(PIN_LAT, HIGH);
  digitalWrite(PIN_LAT, LOW);
}

void drawPixelRunner(uint8_t physicalRow, uint8_t activeCol) {
  digitalWrite(PIN_OE, HIGH); // Disable output

  uint8_t addrRow = physicalRow % SCAN_LINES;  // 0–15
  bool isBottomHalf = physicalRow >= SCAN_LINES;

  setRowAddress(addrRow);

  for (uint8_t x = 0; x < WIDTH; x++) {
    // Set RGB for top and bottom
    digitalWrite(PIN_R1, (!isBottomHalf && x == activeCol) ? HIGH : LOW);
    digitalWrite(PIN_G1, (!isBottomHalf && x == activeCol) ? HIGH : LOW);
    digitalWrite(PIN_B1, (!isBottomHalf && x == activeCol) ? HIGH : LOW);

    digitalWrite(PIN_R2, (isBottomHalf && x == activeCol) ? HIGH : LOW);
    digitalWrite(PIN_G2, (isBottomHalf && x == activeCol) ? HIGH : LOW);
    digitalWrite(PIN_B2, (isBottomHalf && x == activeCol) ? HIGH : LOW);

    pulseClock();
  }

  pulseLatch();
  digitalWrite(PIN_OE, LOW); // Enable output
  delayMicroseconds(WHITE_ON_TIME);
}

// Draws red speed bar (left) and green frame counter (right) on bottom row
void drawSpeedBar() {
  digitalWrite(PIN_OE, HIGH); // Disable output
  uint8_t physicalRow = HEIGHT - 1;
  uint8_t addrRow = physicalRow % SCAN_LINES;
  bool isBottomHalf = physicalRow >= SCAN_LINES;
  setRowAddress(addrRow);
  for (uint8_t x = 0; x < WIDTH; x++) {
    bool redLED = false;
    bool greenLED = false;
    
    if (x < 16) {
      // Left side: red speed LEDs (positions 0-15)
      redLED = (x < pixelRunnerSpeed);
    } else {
      // Right side: green frame counter LEDs (positions 16-31)
      uint8_t bitPosition = 31 - x; // bit 0 at pos 31, bit 15 at pos 16
      greenLED = (frameCounter >> bitPosition) & 1;
    }
    
    digitalWrite(PIN_R1, (!isBottomHalf && redLED) ? HIGH : LOW);
    digitalWrite(PIN_G1, (!isBottomHalf && greenLED) ? HIGH : LOW);
    digitalWrite(PIN_B1, LOW);
    digitalWrite(PIN_R2, (isBottomHalf && redLED) ? HIGH : LOW);
    digitalWrite(PIN_G2, (isBottomHalf && greenLED) ? HIGH : LOW);
    digitalWrite(PIN_B2, LOW);
    pulseClock();
  }
  pulseLatch();
  digitalWrite(PIN_OE, LOW); // Enable output
  delayMicroseconds(BAR_ON_TIME);
}

// Button state tracking for non-blocking debouncing
bool btnUpPressed = false;
bool btnDownPressed = false;
unsigned long lastBtnTime = 0;
const unsigned long DEBOUNCE_DELAY = 150; // ms
uint16_t pixelCounter = 0; // Track position for speed bar updates

void loop() {
  // Non-blocking button reading (only check occasionally)
  unsigned long currentTime = millis();
  if (currentTime - lastBtnTime > DEBOUNCE_DELAY) {
    bool upPressed = (digitalRead(PIN_BTN_UP) == LOW);
    bool downPressed = (digitalRead(PIN_BTN_DOWN) == LOW);
    
    if (upPressed && !btnUpPressed && pixelRunnerSpeed < 16) {
      pixelRunnerSpeed++;
      // Update WHITE_ON_TIME based on speed (higher speed = lower time = faster)
      WHITE_ON_TIME = map(pixelRunnerSpeed, 1, 16, SPEED_MAX_TIME, SPEED_MIN_TIME);
      lastBtnTime = currentTime;
    }
    if (downPressed && !btnDownPressed && pixelRunnerSpeed > 1) {
      pixelRunnerSpeed--;
      // Update WHITE_ON_TIME based on speed (lower speed = higher time = slower)
      WHITE_ON_TIME = map(pixelRunnerSpeed, 1, 16, SPEED_MAX_TIME, SPEED_MIN_TIME);
      lastBtnTime = currentTime;
    }
    
    btnUpPressed = upPressed;
    btnDownPressed = downPressed;
  }

  drawPixelRunner(row, col);
  
  // Only draw speed bar every 32 pixels to minimize timing impact
  if (pixelCounter % 32 == 0) {
    drawSpeedBar();
  }

  col++;
  pixelCounter++;
  if (col >= 32) { // Skip the 32nd column (columns 0-30 only, total 31×32=992 LEDs)
    col = 0;
    row++;
    if (row >= HEIGHT - 1) {
      row = 0;
      frameCounter++; // Increment when completing full sweep
      pixelCounter = 0; // Reset pixel counter
    }
  }

  // Precise 160µs per LED timing - no additional delays
}