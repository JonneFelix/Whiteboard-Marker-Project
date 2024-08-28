#include <Adafruit_BNO08x.h>
#include <SPI.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <SPIFFS.h>

// Pin-Definitionen für BNO08x Sensor und Button
#define BNO08X_CS 9
#define BNO08X_INT 6
#define BNO08X_RESET 5
#define BUTTON_PIN 38 

// BLE-Objekte
BLECharacteristic* pCharacteristic;
BLEServer* pServer;
BLEService* pService;
BLEAdvertising* pAdvertising;
bool deviceConnected = false;

// Hz-Zähler
volatile int packetCount = 0;
unsigned long lastMillis = 0;
bool dataCollectionEnabled = false;

// Sensor object
Adafruit_BNO08x bno08x(BNO08X_RESET);
sh2_SensorValue_t sensorValue;

// Define the report types and intervals
sh2_SensorId_t reportType1 = SH2_LINEAR_ACCELERATION;
sh2_SensorId_t reportType2 = SH2_ROTATION_VECTOR;
long reportIntervalUs = 20000; // 100 Hz

// Filename for calibration data
const char* calibrationFilename = "/calibration.dat";

void setReports(sh2_SensorId_t reportType, long report_interval) {
  Serial.print("Setting report for ");
  Serial.println(reportType);
  if (!bno08x.enableReport(reportType, report_interval)) {
    Serial.print("Could not enable report type: ");
    Serial.println(reportType);
  }
}

void saveCalibrationData() {
  Serial.println("Saving calibration data...");
  int status = sh2_saveDcdNow();
  if (status == SH2_OK) {
    Serial.println("Calibration data saved successfully.");
  } else {
    Serial.print("Failed to save calibration data. Status: ");
    Serial.println(status);
  }
  delay(100); // Small delay to ensure the save command is processed
}

void saveCalibrationToSPIFFS() {
  File file = SPIFFS.open(calibrationFilename, FILE_WRITE);
  if (!file) {
    Serial.println("Failed to open file for writing");
    return;
  }

  // Save a placeholder to indicate calibration has been saved
  file.write((const uint8_t*)"CALIBRATED", 10);
  file.close();
  Serial.println("Calibration data saved to SPIFFS.");
}

bool loadCalibrationData() {
  Serial.println("Loading calibration data...");
  uint8_t sensors;
  int status = sh2_getCalConfig(&sensors);
  if (status == SH2_OK) {
    Serial.println("Calibration data loaded from sensor.");
    return true;
  }
  Serial.print("No calibration data found on sensor. Status: ");
  Serial.println(status);
  return false;
}

void loadCalibrationFromSPIFFS() {
  File file = SPIFFS.open(calibrationFilename, FILE_READ);
  if (!file) {
    Serial.println("Failed to open file for reading");
    return;
  }

  // Check if calibration data exists
  char buffer[11];
  file.readBytes(buffer, 10);
  buffer[10] = '\0';
  file.close();

  if (strcmp(buffer, "CALIBRATED") == 0) {
    Serial.println("Calibration data loaded from SPIFFS.");
  } else {
    Serial.println("No calibration data found in SPIFFS.");
  }
}

void setup() {
  Serial.begin(250000);
  Serial.println("Adafruit BNO08x test!");

  // Initialize SPIFFS
  Serial.println("Initializing SPIFFS...");
  if (!SPIFFS.begin(true)) {
    Serial.println("An error has occurred while mounting SPIFFS");
    while (1) { delay(10); }
  }
  Serial.println("SPIFFS mounted successfully");

  // Initialize BNO08x
  Serial.println("Initializing BNO08x...");
  if (!bno08x.begin_SPI(BNO08X_CS, BNO08X_INT, &SPI, BNO08X_RESET)) {
    Serial.println("Failed to find BNO08x chip");
    while (1) { delay(10); }
  }
  Serial.println("BNO08x Found!");

  // Enable DCD auto-save
  Serial.println("Enabling DCD auto-save...");
  int status = sh2_setDcdAutoSave(true);
  if (status != SH2_OK) {
    Serial.print("Failed to enable DCD auto-save. Status: ");
    Serial.println(status);
  } else {
    Serial.println("DCD auto-save enabled");
  }

  // Load calibration data if available
  loadCalibrationFromSPIFFS();
  if (!loadCalibrationData()) {
    Serial.println("No calibration data found, please calibrate the sensor.");
  } else {
    Serial.println("Calibration data loaded.");
  }

  // Set the reports
  Serial.println("Setting reports...");
  setReports(reportType1, reportIntervalUs);
  setReports(reportType2, reportIntervalUs);

  // Set up BLE
  setupBLE();

  // Set up button pin
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  Serial.println("Reading events");
  delay(100);
}

void controlDataCollection(bool enable) {
  dataCollectionEnabled = enable;
  if (enable) {
    Serial.println("Data Sending enabled");
  } else {
    Serial.println("Data Sending disabled");
  }
}

class MyServerCallbacks : public BLEServerCallbacks {
public:
  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
    Serial.println("BLE Device Connected");
  }

  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
    Serial.println("BLE Device Disconnected");
    controlDataCollection(false);  // Stop data collection if BLE is disconnected
    pServer->startAdvertising();   // Start advertising again
    Serial.println("Suchmodus Aktiv!");
  }
};

void setupBLE() {
  Serial.println("Initializing BLE...");

  BLEDevice::init("BNO085 Sensor");

  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  pService = pServer->createService(BLEUUID("4fafc201-1fb5-459e-8fcc-c5c9c331914b"));

  pCharacteristic = pService->createCharacteristic(
    BLEUUID("beb5483e-36e1-4688-b7f5-ea07361b26a8"),
    BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);

  pCharacteristic->addDescriptor(new BLE2902());

  pService->start();

  pAdvertising = BLEDevice::getAdvertising();
  BLEDevice::setMTU(256);

  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMaxPreferred(0x12);

  Serial.println("BLE Initialized");
  pAdvertising->start();
  Serial.println("Suchmodus Aktiv!");
}

void sendData(float x, float y, float z, float r, float i, float j, float k, float qa) {
  StaticJsonDocument<256> doc;
  doc["x"] = x;
  doc["y"] = y;
  doc["z"] = z;
  doc["r"] = r;
  doc["i"] = i;
  doc["j"] = j;
  doc["k"] = k;
  doc["qa"] = qa;
  doc["collect_data"] = dataCollectionEnabled ? "start" : "stop";

  char jsonBuffer[256];
  size_t n = serializeJson(doc, jsonBuffer);

  pCharacteristic->setValue((uint8_t*)jsonBuffer, n);  
  pCharacteristic->notify();
  packetCount++;
  //Serial.println("Packet count is " + packetCount);
}

void loop() {
  static bool lastButtonState = HIGH;  // Initialer Zustand des Buttons
  bool buttonState = digitalRead(BUTTON_PIN);

  
  if(deviceConnected){ //Überprüft ob device verbunden before 
    // Überprüfen, ob der Button gedrückt wurde (Flankenerkennung)
    if (buttonState == LOW && lastButtonState == HIGH) {
    dataCollectionEnabled = !dataCollectionEnabled;  // Umschalten des Datensendens
    controlDataCollection(dataCollectionEnabled);    // Steuerung des Datensendens
    }
  }else{
    lastButtonState = HIGH;
  }
  
  lastButtonState = buttonState;
  static float x, y, z, r, i, j, k, qa;

  if (bno08x.getSensorEvent(&sensorValue)) {
    switch (sensorValue.sensorId) {
      case SH2_LINEAR_ACCELERATION:
        x = sensorValue.un.linearAcceleration.x;
        y = sensorValue.un.linearAcceleration.y;
        z = sensorValue.un.linearAcceleration.z;
        break;

      case SH2_ROTATION_VECTOR:
        r = sensorValue.un.rotationVector.real;
        i = sensorValue.un.rotationVector.i;
        j = sensorValue.un.rotationVector.j;
        k = sensorValue.un.rotationVector.k;
        qa = sensorValue.status;

          static unsigned long lastSaveTime = 0;
          static bool calibrationSaved = false;

          if (sensorValue.status == 3 && !calibrationSaved) {
            saveCalibrationData();
            saveCalibrationToSPIFFS();
            lastSaveTime = millis();
            calibrationSaved = true;
          } else if (sensorValue.status < 3) {
            calibrationSaved = false;
          }
        break;
    }
    if (deviceConnected) {
      /*Serial.print("Linear Acceleration - x: ");
      Serial.print(x);
      Serial.print(" y: ");
      Serial.print(y);
      Serial.print(" z: ");
      Serial.print(z);
      Serial.print(" | ");
      Serial.print("Rotation Vector - r: ");
      Serial.print(r);
      Serial.print(" i: ");
      Serial.print(i);
      Serial.print(" j: ");
      Serial.print(j);
      Serial.print(" k: ");
      Serial.print(k);
      Serial.print(" | ");
      Serial.println(qa);  // Print accuracy
      */
      sendData(x, y, z, r, i, j, k, qa);
    }
    
  }

  if (millis() - lastMillis >= 1000) {
    if (deviceConnected) {
      Serial.print("Send rate: ");
      Serial.print(packetCount);
      Serial.println(" Hz");
      packetCount = 0;
    }
    lastMillis = millis();
  }
}




