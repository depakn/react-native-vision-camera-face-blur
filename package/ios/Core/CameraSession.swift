//
//  CameraSession.swift
//  VisionCamera
//
//  Created by Marc Rousavy on 11.10.23.
//  Copyright © 2023 mrousavy. All rights reserved.
//

import AVFoundation
import Foundation
import os.log
import UIKit
import Vision

/// A fully-featured Camera Session supporting preview, video, photo, frame processing, and code scanning outputs.
/// All changes to the session have to be controlled via the `configure` function.
final class CameraSession: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate,
  AVCaptureAudioDataOutputSampleBufferDelegate {
  // Configuration
  private var isInitialized = false
  var configuration: CameraConfiguration?
  var currentConfigureCall: DispatchTime = .now()
  // Capture Session
  let captureSession = AVCaptureSession()
  let audioCaptureSession = AVCaptureSession()
  // Inputs & Outputs
  var videoDeviceInput: AVCaptureDeviceInput?
  var audioDeviceInput: AVCaptureDeviceInput?
  var photoOutput: AVCapturePhotoOutput?
  var videoOutput: AVCaptureVideoDataOutput?
  var audioOutput: AVCaptureAudioDataOutput?
  var codeScannerOutput: AVCaptureMetadataOutput?
  // State
  var metadataProvider = MetadataProvider()
  var recordingSession: RecordingSession?
  var didCancelRecording = false
  var orientationManager = OrientationManager()

  // Callbacks
  weak var delegate: CameraSessionDelegate?

  // Public accessors
  var maxZoom: Double {
    if let device = videoDeviceInput?.device {
      return device.activeFormat.videoMaxZoomFactor
    }
    return 1.0
  }

  private let faceDetector: VNSequenceRequestHandler
  private let ciContext: CIContext
  private let logger = OSLog(subsystem: "com.sdtech.rnvce", category: "CameraSession")

  private let isDebugMode = true
  private var frameCount = 0
  private var videoSize: CGSize?

  /**
   Create a new instance of the `CameraSession`.
   The `onError` callback is used for any runtime errors.
   */
  override init() {
    faceDetector = VNSequenceRequestHandler()
    ciContext = CIContext()

    super.init()
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(sessionRuntimeError),
      name: .AVCaptureSessionRuntimeError,
      object: captureSession)
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(sessionRuntimeError),
      name: .AVCaptureSessionRuntimeError,
      object: audioCaptureSession)
    NotificationCenter.default.addObserver(
      self,
      selector: #selector(audioSessionInterrupted),
      name: AVAudioSession.interruptionNotification,
      object: AVAudioSession.sharedInstance)
  }

  private func initialize() {
    if isInitialized {
      return
    }
    orientationManager.delegate = self
    isInitialized = true
  }

  deinit {
    NotificationCenter.default.removeObserver(
      self,
      name: .AVCaptureSessionRuntimeError,
      object: captureSession)
    NotificationCenter.default.removeObserver(
      self,
      name: .AVCaptureSessionRuntimeError,
      object: audioCaptureSession)
    NotificationCenter.default.removeObserver(
      self,
      name: AVAudioSession.interruptionNotification,
      object: AVAudioSession.sharedInstance)
  }

  /**
   Creates a PreviewView for the current Capture Session
   */
  func createPreviewView(frame: CGRect) -> PreviewView {
    return PreviewView(frame: frame, session: captureSession)
  }

  func onConfigureError(_ error: Error) {
    if let error = error as? CameraError {
      // It's a typed Error
      delegate?.onError(error)
    } else {
      // It's any kind of unknown error
      let cameraError = CameraError.unknown(message: error.localizedDescription)
      delegate?.onError(cameraError)
    }
  }

  /**
   Update the session configuration.
   Any changes in here will be re-configured only if required, and under a lock (in this case, the serial cameraQueue DispatchQueue).
   The `configuration` object is a copy of the currently active configuration that can be modified by the caller in the lambda.
   */
  func configure(_ lambda: @escaping (_ configuration: CameraConfiguration) throws -> Void) {
    initialize()

    VisionLogger.log(level: .info, message: "configure { ... }: Waiting for lock...")

    // Set up Camera (Video) Capture Session (on camera queue, acts like a lock)
    CameraQueues.cameraQueue.async {
      // Let caller configure a new configuration for the Camera.
      let config = CameraConfiguration(copyOf: self.configuration)
      do {
        try lambda(config)
      } catch CameraConfiguration.AbortThrow.abort {
        // call has been aborted and changes shall be discarded
        return
      } catch {
        // another error occured, possibly while trying to parse enums
        self.onConfigureError(error)
        return
      }
      let difference = CameraConfiguration.Difference(between: self.configuration, and: config)

      VisionLogger.log(
        level: .info,
        message: "configure { ... }: Updating CameraSession Configuration... \(difference)")

      do {
        // If needed, configure the AVCaptureSession (inputs, outputs)
        if difference.isSessionConfigurationDirty {
          self.captureSession.beginConfiguration()

          // 1. Update input device
          if difference.inputChanged {
            try self.configureDevice(configuration: config)
          }
          // 2. Update outputs
          if difference.outputsChanged {
            try self.configureOutputs(configuration: config)
          }
          // 3. Update Video Stabilization
          if difference.videoStabilizationChanged {
            self.configureVideoStabilization(configuration: config)
          }
          // 4. Update target output orientation
          if difference.orientationChanged {
            self.orientationManager.setTargetOutputOrientation(config.outputOrientation)
          }
        }

        guard let device = self.videoDeviceInput?.device else {
          throw CameraError.device(.noDevice)
        }

        // If needed, configure the AVCaptureDevice (format, zoom, low-light-boost, ..)
        if difference.isDeviceConfigurationDirty {
          try device.lockForConfiguration()
          defer {
            device.unlockForConfiguration()
          }

          // 5. Configure format
          if difference.formatChanged {
            try self.configureFormat(configuration: config, device: device)
          }
          // 6. After step 2. and 4., we also need to configure some output properties that depend on format.
          //    This needs to be done AFTER we updated the `format`, as this controls the supported properties.
          if difference.outputsChanged || difference.formatChanged {
            self.configureVideoOutputFormat(configuration: config)
            self.configurePhotoOutputFormat(configuration: config)
          }
          // 7. Configure side-props (fps, lowLightBoost)
          if difference.sidePropsChanged {
            try self.configureSideProps(configuration: config, device: device)
          }
          // 8. Configure zoom
          if difference.zoomChanged {
            self.configureZoom(configuration: config, device: device)
          }
          // 9. Configure exposure bias
          if difference.exposureChanged {
            self.configureExposure(configuration: config, device: device)
          }
        }

        if difference.isSessionConfigurationDirty {
          // We commit the session config updates AFTER the device config,
          // that way we can also batch those changes into one update instead of doing two updates.
          self.captureSession.commitConfiguration()
        }

        // 10. Start or stop the session if needed
        self.checkIsActive(configuration: config)

        // 11. Enable or disable the Torch if needed (requires session to be running)
        if difference.torchChanged {
          try device.lockForConfiguration()
          defer {
            device.unlockForConfiguration()
          }
          try self.configureTorch(configuration: config, device: device)
        }

        // After configuring, set this to the new configuration.
        self.configuration = config
      } catch {
        self.onConfigureError(error)
      }

      // Set up Audio Capture Session (on audio queue)
      if difference.audioSessionChanged {
        CameraQueues.audioQueue.async {
          do {
            // Lock Capture Session for configuration
            VisionLogger.log(level: .info, message: "Beginning AudioSession configuration...")
            self.audioCaptureSession.beginConfiguration()

            try self.configureAudioSession(configuration: config)

            // Unlock Capture Session again and submit configuration to Hardware
            self.audioCaptureSession.commitConfiguration()
            VisionLogger.log(level: .info, message: "Committed AudioSession configuration!")
          } catch {
            self.onConfigureError(error)
          }
        }
      }

      // Set up Location streaming (on location queue)
      if difference.locationChanged {
        CameraQueues.locationQueue.async {
          do {
            VisionLogger.log(level: .info, message: "Beginning Location Output configuration...")
            try self.configureLocationOutput(configuration: config)
            VisionLogger.log(level: .info, message: "Finished Location Output configuration!")
          } catch {
            self.onConfigureError(error)
          }
        }
      }
    }
  }

  /**
   Starts or stops the CaptureSession if needed (`isActive`)
   */
  private func checkIsActive(configuration: CameraConfiguration) {
    if configuration.isActive == captureSession.isRunning {
      return
    }

    // Start/Stop session
    if configuration.isActive {
      captureSession.startRunning()
      delegate?.onCameraStarted()
    } else {
      captureSession.stopRunning()
      delegate?.onCameraStopped()
    }
  }

  public final func captureOutput(
    _ captureOutput: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    switch captureOutput {
    case is AVCaptureVideoDataOutput:
      onVideoFrame(
        sampleBuffer: sampleBuffer, orientation: connection.orientation,
        isMirrored: connection.isVideoMirrored)
    case is AVCaptureAudioDataOutput:
      onAudioFrame(sampleBuffer: sampleBuffer)
    default:
      break
    }
  }

  private final func onVideoFrame(
    sampleBuffer: CMSampleBuffer, orientation: Orientation, isMirrored: Bool
  ) {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
      os_log("Failed to get pixel buffer from sample buffer", log: logger, type: .error)
      return
    }

    var ciImage = CIImage(cvPixelBuffer: pixelBuffer)

    // Apply orientation correction
    ciImage = ciImage.oriented(orientation.cgImagePropertyOrientation)

    // Detect faces
    let faceDetectionRequest = VNDetectFaceLandmarksRequest()
    do {
      try faceDetector.perform([faceDetectionRequest], on: ciImage, orientation: .up)
    } catch {
      os_log(
        "Face detection failed: %{public}@", log: logger, type: .error, error.localizedDescription)
    }

    let faces = faceDetectionRequest.results as? [VNFaceObservation] ?? []
    os_log("Detected %d faces", log: logger, type: .debug, faces.count)

    // Blur detected faces
    let blurredImage = applyFaceBlur(to: ciImage, faces: faces)

    if isDebugMode {
      frameCount += 1
      if frameCount % 60 == 0 {  // Save every 60th frame
        saveImageForDebug(ciImage: blurredImage, prefix: "blurred", orientation: orientation)
        saveImageForDebug(ciImage: ciImage, prefix: "original", orientation: orientation)

        // Log face detection details
        for (index, face) in faces.enumerated() {
          let faceBounds = face.boundingBox
          os_log(
            "Face %d detected at normalized rect: {{%f, %f}, {%f, %f}}", log: logger, type: .debug,
            index, faceBounds.origin.x, faceBounds.origin.y, faceBounds.size.width,
            faceBounds.size.height)
        }
      }
    }

    // Convert back to CMSampleBuffer
    guard
      let blurredBuffer = blurredImage.toCMSampleBuffer(
        from: sampleBuffer, ciContext: self.ciContext)
    else {
      os_log("Failed to convert blurred image to sample buffer", log: logger, type: .error)
      return
    }

    if let recordingSession {
      do {
        // Write the blurred Video Buffer to the RecordingSession
        try recordingSession.append(buffer: blurredBuffer, ofType: .video)
      } catch {
        os_log(
          "Recording failed: %{public}@", log: logger, type: .error, error.localizedDescription)
        delegate?.onError(.capture(.unknown(message: error.localizedDescription)))
      }
    }

    if let delegate {
      // Call Frame Processor (delegate) for every Video Frame
      delegate.onFrame(
        sampleBuffer: blurredBuffer, orientation: orientation, isMirrored: isMirrored)
    }
  }

  private func applyFaceBlur(to image: CIImage, faces: [VNFaceObservation]) -> CIImage {
    guard !faces.isEmpty else {
      os_log("No faces detected, returning original image", log: logger, type: .debug)
      return image
    }

    var outputImage = image
    let imageSize = image.extent.size

    for (index, face) in faces.enumerated() {
      // Convert normalized coordinates to pixel coordinates
      let faceBounds = VNImageRectForNormalizedRect(
        face.boundingBox, Int(imageSize.width), Int(imageSize.height))

      // Add padding to face bounds (10% on each side)
      let paddedFaceBounds = faceBounds.insetBy(
        dx: -faceBounds.width * 0.1, dy: -faceBounds.height * 0.1)

      // Ensure the blur area is within the image bounds
      let constrainedFaceBounds = paddedFaceBounds.intersection(image.extent)

      let blurFilter = CIFilter(name: "CIGaussianBlur")!
      blurFilter.setValue(99.0, forKey: kCIInputRadiusKey)
      blurFilter.setValue(image.cropped(to: constrainedFaceBounds), forKey: kCIInputImageKey)

      if let blurredFace = blurFilter.outputImage {
        // Create a masked blur effect
        let maskImage = CIImage(color: .white).cropped(to: constrainedFaceBounds)
        let maskedBlur = blurredFace.applyingFilter(
          "CIBlendWithMask",
          parameters: [
            kCIInputBackgroundImageKey: image.cropped(to: constrainedFaceBounds),
            kCIInputMaskImageKey: maskImage,
          ])

        // Composite the blurred face onto the original image
        outputImage = maskedBlur.composited(over: outputImage)

        os_log(
          "Applied blur to face %d at rect: %@", log: logger, type: .debug, index,
          NSCoder.string(for: constrainedFaceBounds))
      } else {
        os_log("Failed to apply blur to face %d", log: logger, type: .error, index)
      }
    }

    return outputImage
  }

  private func saveImageForDebug(ciImage: CIImage, prefix: String, orientation: Orientation) {
    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "yyyyMMdd_HHmmss_SSS"
    let timestamp = dateFormatter.string(from: Date())

    guard
      let documentsDirectory = FileManager.default.urls(
        for: .documentDirectory, in: .userDomainMask
      ).first
    else {
      os_log("Failed to get documents directory", log: logger, type: .error)
      return
    }

    let fileName = "\(prefix)_\(timestamp).jpg"
    let fileURL = documentsDirectory.appendingPathComponent(fileName)

    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
      os_log("Failed to create color space", log: logger, type: .error)
      return
    }

    guard
      let cgImage = ciContext.createCGImage(
        ciImage, from: ciImage.extent, format: .RGBA8, colorSpace: colorSpace)
    else {
      os_log("Failed to create CGImage", log: logger, type: .error)
      return
    }

    let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: orientation.uiImageOrientation)

    guard let data = uiImage.jpegData(compressionQuality: 0.8) else {
      os_log("Failed to create JPEG data", log: logger, type: .error)
      return
    }

    do {
      try data.write(to: fileURL)
      os_log("Saved debug image: %@", log: logger, type: .debug, fileURL.path)
    } catch {
      os_log(
        "Failed to save debug image: %@", log: logger, type: .error, error.localizedDescription)
    }
  }

  private final func onAudioFrame(sampleBuffer: CMSampleBuffer) {
    if let recordingSession {
      do {
        // Synchronize the Audio Buffer with the Video Session's time because it's two separate
        // AVCaptureSessions, then write it to the .mov/.mp4 file
        audioCaptureSession.synchronizeBuffer(sampleBuffer, toSession: captureSession)
        try recordingSession.append(buffer: sampleBuffer, ofType: .audio)
      } catch let error as CameraError {
        delegate?.onError(error)
      } catch {
        delegate?.onError(.capture(.unknown(message: error.localizedDescription)))
      }
    }
  }

  // pragma MARK: Notifications

  @objc
  func sessionRuntimeError(notification: Notification) {
    VisionLogger.log(level: .error, message: "Unexpected Camera Runtime Error occured!")
    guard let error = notification.userInfo?[AVCaptureSessionErrorKey] as? AVError else {
      return
    }

    // Notify consumer about runtime error
    delegate?.onError(.unknown(message: error._nsError.description, cause: error._nsError))

    let shouldRestart = configuration?.isActive == true
    if shouldRestart {
      // restart capture session after an error occured
      CameraQueues.cameraQueue.async {
        self.captureSession.startRunning()
      }
    }
  }
}

extension Orientation {
  var cgImagePropertyOrientation: CGImagePropertyOrientation {
    switch self {
    case .portrait: return .right
    case .portraitUpsideDown: return .left
    case .landscapeLeft: return .down
    case .landscapeRight: return .up
    }
  }

  var uiImageOrientation: UIImage.Orientation {
    switch self {
    case .portrait: return .right
    case .portraitUpsideDown: return .left
    case .landscapeLeft: return .down
    case .landscapeRight: return .up
    }
  }
}

extension CIImage {
  func toCMSampleBuffer(from sampleBuffer: CMSampleBuffer, ciContext: CIContext) -> CMSampleBuffer? {
    var pixelBuffer: CVPixelBuffer?
    let attrs =
      [
        kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
        kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue,
        kCVPixelBufferIOSurfacePropertiesKey: [:],
      ] as CFDictionary

    CVPixelBufferCreate(
      kCFAllocatorDefault,
      Int(extent.width),
      Int(extent.height),
      kCVPixelFormatType_32BGRA,
      attrs,
      &pixelBuffer)

    guard let pixelBuffer = pixelBuffer else { return nil }

    ciContext.render(self, to: pixelBuffer)

    var newSampleBuffer: CMSampleBuffer?
    var timingInfo = CMSampleTimingInfo()
    CMSampleBufferGetSampleTimingInfo(sampleBuffer, at: 0, timingInfoOut: &timingInfo)

    var videoInfo: CMVideoFormatDescription?
    CMVideoFormatDescriptionCreateForImageBuffer(
      allocator: kCFAllocatorDefault, imageBuffer: pixelBuffer, formatDescriptionOut: &videoInfo)

    CMSampleBufferCreateForImageBuffer(
      allocator: kCFAllocatorDefault,
      imageBuffer: pixelBuffer,
      dataReady: true,
      makeDataReadyCallback: nil,
      refcon: nil,
      formatDescription: videoInfo!,
      sampleTiming: &timingInfo,
      sampleBufferOut: &newSampleBuffer)

    return newSampleBuffer
  }
}
