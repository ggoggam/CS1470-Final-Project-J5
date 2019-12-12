//
//  ViewController.swift
//  Generative Improv-Theremin
//
//  Created by Jaehyun Jeon on 12/11/19.
//  Copyright © 2019 Jaehyun Jeon. All rights reserved.
//

import UIKit
import AudioUnit
import CoreAudioKit
import CoreAudio
import AudioToolbox
import AVFoundation
import Foundation

class ViewController: UIViewController {
    
    var processingGraph: AUGraph?
    var midisynthNode   = AUNode()
    var ioNode          = AUNode()
    var midisynthUnit: AudioUnit?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
//        initAudio()
        
        newAudio()
        loadPatch2(gmpatch: 0)
        
        
        
    }
    
    var engine = AVAudioEngine()
    var sampler = AVAudioUnitSampler()
    
    func newAudio() {
//        engine = AVAudioEngine()
//        sampler = AVAudioUnitSampler()
        engine.attach(sampler)
        engine.connect(sampler, to: engine.mainMixerNode, format: nil)
        engine = AVAudioEngine()
        sampler = AVAudioUnitSampler()
        engine.attach(sampler)
        engine.connect(sampler, to: engine.mainMixerNode, format: nil)
        
        do {
            try engine.start()
            print("audio engine started")
        } catch {
            print("oops \(error)")
            print("could not start audio engine")
        }
        
    }
    
    // instance variables
    let melodicBank = UInt8(kAUSampler_DefaultMelodicBankMSB)
    let defaultBankLSB = UInt8(kAUSampler_DefaultBankLSB)
    let gmMarimba = UInt8(12)
    let gmHarpsichord = UInt8(6)

    func loadPatch2(gmpatch:UInt8, channel:UInt8 = 0) {
            
        guard let soundbank =
            Bundle.main.url(forResource: "GeneralUser GS MuseScore v1.442", withExtension: "sf2")
            else {
                print("could not read sound font")
                return
        }
            
        do {
            try sampler.loadSoundBankInstrument(at: soundbank, program:gmpatch,
                bankMSB: melodicBank, bankLSB: defaultBankLSB)
                
        } catch let error as NSError {
            print("\(error.localizedDescription)")
            return
        }
            
        self.sampler.sendProgramChange(gmpatch, bankMSB: melodicBank, bankLSB: defaultBankLSB, onChannel: channel)
    }
    
    func initAudio() {
        
        // the graph is like a patch bay, where everything gets connected
        
        
        CheckError(NewAUGraph(&processingGraph))
        
        
        createIONode()
        createSynthNode()
        CheckError(AUGraphOpen(processingGraph!))
        CheckError(AUGraphNodeInfo(processingGraph!, midisynthNode, nil, &midisynthUnit))
        let synthOutputElement:AudioUnitElement = 0
        let ioUnitInputElement:AudioUnitElement = 0
        CheckError(AUGraphConnectNodeInput(processingGraph!, midisynthNode, synthOutputElement, ioNode, ioUnitInputElement))
        
        
        
        
        CheckError(AUGraphInitialize(processingGraph!))
        
        
        
        
        CheckError(AUGraphStart(processingGraph!))
        
//        loadVoices()
    }
    
    private func createIONode() {
        var cd = AudioComponentDescription(
            componentType: kAudioUnitType_Output,
            componentSubType: kAudioUnitSubType_RemoteIO,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,componentFlagsMask: 0)
        CheckError(AUGraphAddNode(processingGraph!, &cd, &ioNode))
    }
    
    private func createSynthNode() {
        var cd = AudioComponentDescription(
            componentType: kAudioUnitType_MusicDevice,
            componentSubType: kAudioUnitSubType_Sampler,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,componentFlagsMask: 0)
        CheckError(AUGraphAddNode(processingGraph!, &cd, &midisynthNode))
    }
    
    func noteOn(note:Int) {
        print(note)
//        let midiChannel = UInt32(0)
//        let noteCommand = UInt32(0x90 | midiChannel)
        //       let base = note - 48
        //       let octaveAdjust = (UInt8(octave) * 12) + base
//        let pitch = UInt32(note)
//        DispatchQueue.main.async {
        self.sampler.startNote(UInt8(note), withVelocity: 50, onChannel: 0)
//        }
        
//        CheckError(MusicDeviceMIDIEvent(self.midisynthUnit!,
//                                        noteCommand, pitch, UInt32(100), 0))
    }
    
    func noteOff(note:Int) {
//        let channel = UInt32(0)
//        let noteCommand = UInt32(0x80 | channel)
        //      let base = note - 48
        //      let octaveAdjust = (UInt8(octave) * 12) + base
        //      let pitch = UInt32(octaveAdjust)
//        let pitch = UInt32(note)
        
        
        self.sampler.stopNote(UInt8(note), onChannel: 0)
//        CheckError(MusicDeviceMIDIEvent(self.midisynthUnit!,
//                                        noteCommand, pitch, 0, 0))
    }
    
    var currentPitchPerButton: [Int: Int] = [:]
    var currentlyON: [Int:Bool] = [:]
    
    @IBAction func onButton0(_ sender: Any) {
//        self.noteOn(note: 50)
//        currentPitchPerButton[0] = 50
        getNoteFromButton(button: 0)
    }
    @IBAction func onButton1(_ sender: Any) {
        getNoteFromButton(button: 1)
    }
    @IBAction func onButton2(_ sender: Any) {
        getNoteFromButton(button: 2)
    }
    @IBAction func onButton3(_ sender: Any) {
        getNoteFromButton(button: 3)
    }
    @IBAction func onButton4(_ sender: Any) {
        getNoteFromButton(button: 4)
    }
    @IBAction func onButton5(_ sender: Any) {
        getNoteFromButton(button: 5)
    }
    
    @IBAction func onButton6(_ sender: Any) {
        getNoteFromButton(button: 6)
    }
    @IBAction func onButton7(_ sender: Any) {
        getNoteFromButton(button: 7)
    }
    
    @IBAction func offButton0(_ sender: Any) {
//        DispatchQueue.main.async {
//            let myNote:Int = self.currentPitchPerButton[0] ?? 21
//        self.noteOff(note: myNote)
//        }
        
    }
    
    @IBAction func offButton1(_ sender: Any) {
//        DispatchQueue.main.async {
//            let myNote:Int = self.currentPitchPerButton[1] ?? 21
//        self.noteOff(note: myNote)
//        }
    }
    
    @IBAction func offButton2(_ sender: Any) {
//        DispatchQueue.main.async {
//            let myNote:Int = self.currentPitchPerButton[2] ?? 21
//        self.noteOff(note: myNote)
//        }
    }
    
    @IBAction func offButton3(_ sender: Any) {
//        DispatchQueue.main.async {
//            let myNote:Int = self.currentPitchPerButton[3] ?? 21
//        self.noteOff(note: myNote)
//        }
    }
    
    @IBAction func offButton4(_ sender: Any) {
//        DispatchQueue.main.async {
//            let myNote:Int = self.currentPitchPerButton[4] ?? 21
//        self.noteOff(note: myNote)
//        }
    }
    
    @IBAction func offButton5(_ sender: Any) {
//        DispatchQueue.main.async {
//            let myNote:Int = self.currentPitchPerButton[5] ?? 21
//        self.noteOff(note: myNote)
//        }
    }
    
    @IBAction func offButton6(_ sender: Any) {
//        DispatchQueue.main.async {
//            let myNote:Int = self.currentPitchPerButton[6] ?? 21
//        self.noteOff(note: myNote)
//        }
    }
    
    @IBAction func offButton7(_ sender: Any) {
//        DispatchQueue.main.async {
//            let myNote:Int = self.currentPitchPerButton[7] ?? 21
//        self.noteOff(note: myNote)
//        }
    }
    
    func getNoteFromButton(button: Int) {
        let url = URL(string: "http://10.38.48.112:5000/\(button)")!
//        DispatchQueue.global(qos: .background).async{
        let task = URLSession.shared.dataTask(with: url) { (data, response, error) in
            if let error = error {
                print("error: \(error)")
            } else {
                
//                if let response = response as? HTTPURLResponse {
//                    //                    print("statusCode: \(response.statusCode)")
//                }
                if let data = data, let dataString = String(data: data, encoding: .utf8) {
                    let note = Int(dataString)
                    
                    DispatchQueue.main.async {
//                        self.noteOff(note: self.currentPitchPerButton[button] ?? 21)
//                      self.currentPitchPerButton[button] = note
                    self.noteOn(note: note ?? 21)
                    }
                    
                }
            }
        }
        task.resume()
//        }
    }
    
    func loadSoundFont() {
//      var bankURL = Bundle.main.url(forResource: "GeneralUser GS MuseScore v1.442",
//        withExtension: "sf2")
        guard var soundbank =
            Bundle.main.url(forResource: "GeneralUser GS MuseScore v1.442", withExtension: "sf2")
            else {
                print("could not read sound font")
                return
        }
//        CheckError(AudioUnitSetProperty(self.midisynthUnit!,
//        kMusicDeviceProperty_SoundBankURL,
//        kAudioUnitScope_Global,
//                                                0,
//                                                &soundbank,
//                                                UInt32(MemoryLayout.size(ofValue: soundbank))))
        
        let cfurl = soundbank as CFURL
        var um = Unmanaged<CFURL>.passRetained(cfurl)
        
        var instdata = AUSamplerInstrumentData(fileURL: um, instrumentType: UInt8(kInstrumentType_DLSPreset), bankMSB: UInt8(kAUSampler_DefaultMelodicBankMSB), bankLSB: UInt8(kAUSampler_DefaultBankLSB), presetID: 0)
//           instdata.bankURL  = (CFURLRef) bankURL;
//           instdata.instrumentType = kInstrumentType_DLSPreset;
//           instdata.bankMSB  = kAUSampler_DefaultMelodicBankMSB;
//           instdata.bankLSB  = kAUSampler_DefaultBankLSB;
//           instdata.presetID = (UInt8) presetNumber;
//
           // set the kAUSamplerProperty_LoadPresetFromBank property
           let result = AudioUnitSetProperty(self.midisynthUnit!,
                                         kAUSamplerProperty_LoadInstrument,
                                         kAudioUnitScope_Global,
                                         0,
                                         &instdata,
                                         UInt32(MemoryLayout.size(ofValue: instdata)))
        
    }
    
    func loadPatch(patchNo: Int) {
      let channel = UInt32(0)
      var enabled = UInt32(1)
      var disabled = UInt32(0)
      
      let patch1 = UInt32(patchNo)
      
      CheckError(AudioUnitSetProperty(
        midisynthUnit!,
        AudioUnitPropertyID(kAUMIDISynthProperty_EnablePreload),
        AudioUnitScope(kAudioUnitScope_Global),
        0,
        &enabled,
        UInt32(MemoryLayout.size(ofValue: enabled))))
      
      let programChangeCommand = UInt32(0xC0 | channel)
      CheckError(MusicDeviceMIDIEvent(midisynthUnit!,
        programChangeCommand, patch1, 0, 0))
      
      CheckError(AudioUnitSetProperty(
        midisynthUnit!,
        AudioUnitPropertyID(kAUMIDISynthProperty_EnablePreload),
        AudioUnitScope(kAudioUnitScope_Global),
        0,
        &disabled,
        UInt32(MemoryLayout<UInt32>.size)))

      // the previous programChangeCommand just triggered a preload
      // this one actually changes to the new voice
      CheckError(MusicDeviceMIDIEvent(midisynthUnit!,
          programChangeCommand, patch1, 0, 0))
    }
    
    func loadVoices() {
        self.loadSoundFont()
        self.loadPatch(patchNo: 0)
//      DispatchQueue.global(qos: .background).async {
//
////        DispatchQueue.main.async {
////          // don't let the user choose a voice until they finish loading
////          self.voiceSelectorView.setShowVoices(show: true)
////          // don't let the user use the sequencer until the voices are loaded
////          self.setUpSequencer()
////        }
//      }
    }
    
    
    
    func CheckError(_ error: OSStatus) {
//                #if os(tvOS) // No CoreMIDI
//                    switch error {
//                    case noErr:
//                        return
//                    case kAudio_ParamError:
//                        AKLog("Error: kAudio_ParamError \n")
//
//                    case kAUGraphErr_NodeNotFound:
//                        AKLog("Error: kAUGraphErr_NodeNotFound \n")
//
//                    case kAUGraphErr_OutputNodeErr:
//                        AKLog( "Error: kAUGraphErr_OutputNodeErr \n")
//
//                    case kAUGraphErr_InvalidConnection:
//                        AKLog("Error: kAUGraphErr_InvalidConnection \n")
//
//                    case kAUGraphErr_CannotDoInCurrentContext:
//                        AKLog( "Error: kAUGraphErr_CannotDoInCurrentContext \n")
//
//                    case kAUGraphErr_InvalidAudioUnit:
//                        AKLog( "Error: kAUGraphErr_InvalidAudioUnit \n")
//
//                    case kAudioToolboxErr_InvalidSequenceType :
//                        AKLog( "Error: kAudioToolboxErr_InvalidSequenceType ")
//
//                    case kAudioToolboxErr_TrackIndexError :
//                        AKLog( "Error: kAudioToolboxErr_TrackIndexError ")
//
//                    case kAudioToolboxErr_TrackNotFound :
//                        AKLog( "Error: kAudioToolboxErr_TrackNotFound ")
//
//                    case kAudioToolboxErr_EndOfTrack :
//                        AKLog( "Error: kAudioToolboxErr_EndOfTrack ")
//
//                    case kAudioToolboxErr_StartOfTrack :
//                        AKLog( "Error: kAudioToolboxErr_StartOfTrack ")
//
//                    case kAudioToolboxErr_IllegalTrackDestination :
//                        AKLog( "Error: kAudioToolboxErr_IllegalTrackDestination")
//
//                    case kAudioToolboxErr_NoSequence :
//                        AKLog( "Error: kAudioToolboxErr_NoSequence ")
//
//                    case kAudioToolboxErr_InvalidEventType :
//                        AKLog( "Error: kAudioToolboxErr_InvalidEventType")
//
//                    case kAudioToolboxErr_InvalidPlayerState :
//                        AKLog( "Error: kAudioToolboxErr_InvalidPlayerState")
//
//                    case kAudioUnitErr_InvalidProperty :
//                        AKLog( "Error: kAudioUnitErr_InvalidProperty")
//
//                    case kAudioUnitErr_InvalidParameter :
//                        AKLog( "Error: kAudioUnitErr_InvalidParameter")
//
//                    case kAudioUnitErr_InvalidElement :
//                        AKLog( "Error: kAudioUnitErr_InvalidElement")
//
//                    case kAudioUnitErr_NoConnection :
//                        AKLog( "Error: kAudioUnitErr_NoConnection")
//
//                    case kAudioUnitErr_FailedInitialization :
//                        AKLog( "Error: kAudioUnitErr_FailedInitialization")
//
//                    case kAudioUnitErr_TooManyFramesToProcess :
//                        AKLog( "Error: kAudioUnitErr_TooManyFramesToProcess")
//
//                    case kAudioUnitErr_InvalidFile :
//                        AKLog( "Error: kAudioUnitErr_InvalidFile")
//
//                    case kAudioUnitErr_FormatNotSupported :
//                        AKLog( "Error: kAudioUnitErr_FormatNotSupported")
//
//                    case kAudioUnitErr_Uninitialized :
//                        AKLog( "Error: kAudioUnitErr_Uninitialized")
//
//                    case kAudioUnitErr_InvalidScope :
//                        AKLog( "Error: kAudioUnitErr_InvalidScope")
//
//                    case kAudioUnitErr_PropertyNotWritable :
//                        AKLog( "Error: kAudioUnitErr_PropertyNotWritable")
//
//                    case kAudioUnitErr_InvalidPropertyValue :
//                        AKLog( "Error: kAudioUnitErr_InvalidPropertyValue")
//
//                    case kAudioUnitErr_PropertyNotInUse :
//                        AKLog( "Error: kAudioUnitErr_PropertyNotInUse")
//
//                    case kAudioUnitErr_Initialized :
//                        AKLog( "Error: kAudioUnitErr_Initialized")
//
//                    case kAudioUnitErr_InvalidOfflineRender :
//                        AKLog( "Error: kAudioUnitErr_InvalidOfflineRender")
//
//                    case kAudioUnitErr_Unauthorized :
//                        AKLog( "Error: kAudioUnitErr_Unauthorized")
//
//                    default:
//                        AKLog("Error: \(error)")
//                    }
//                #else
//                    switch error {
//                    case noErr:
//                        return
//                    case kAudio_ParamError:
//                        AKLog("Error: kAudio_ParamError \n")
//
//                    case kAUGraphErr_NodeNotFound:
//                        AKLog("Error: kAUGraphErr_NodeNotFound \n")
//
//                    case kAUGraphErr_OutputNodeErr:
//                        AKLog( "Error: kAUGraphErr_OutputNodeErr \n")
//
//                    case kAUGraphErr_InvalidConnection:
//                        AKLog("Error: kAUGraphErr_InvalidConnection \n")
//
//                    case kAUGraphErr_CannotDoInCurrentContext:
//                        AKLog( "Error: kAUGraphErr_CannotDoInCurrentContext \n")
//
//                    case kAUGraphErr_InvalidAudioUnit:
//                        AKLog( "Error: kAUGraphErr_InvalidAudioUnit \n")
//
//                    case kMIDIInvalidClient :
//                        AKLog( "kMIDIInvalidClient ")
//
//                    case kMIDIInvalidPort :
//                        AKLog( "Error: kMIDIInvalidPort ")
//
//                    case kMIDIWrongEndpointType :
//                        AKLog( "Error: kMIDIWrongEndpointType")
//
//                    case kMIDINoConnection :
//                        AKLog( "Error: kMIDINoConnection ")
//
//                    case kMIDIUnknownEndpoint :
//                        AKLog( "Error: kMIDIUnknownEndpoint ")
//
//                    case kMIDIUnknownProperty :
//                        AKLog( "Error: kMIDIUnknownProperty ")
//
//                    case kMIDIWrongPropertyType :
//                        AKLog( "Error: kMIDIWrongPropertyType ")
//
//                    case kMIDINoCurrentSetup :
//                        AKLog( "Error: kMIDINoCurrentSetup ")
//
//                    case kMIDIMessageSendErr :
//                        AKLog( "kError: MIDIMessageSendErr ")
//
//                    case kMIDIServerStartErr :
//                        AKLog( "kError: MIDIServerStartErr ")
//
//                    case kMIDISetupFormatErr :
//                        AKLog( "Error: kMIDISetupFormatErr ")
//
//                    case kMIDIWrongThread :
//                        AKLog( "Error: kMIDIWrongThread ")
//
//                    case kMIDIObjectNotFound :
//                        AKLog( "Error: kMIDIObjectNotFound ")
//
//                    case kMIDIIDNotUnique :
//                        AKLog( "Error: kMIDIIDNotUnique ")
//
//                    case kMIDINotPermitted:
//                        AKLog( "Error: kMIDINotPermitted: Have you enabled the audio background mode in your ios app?")
//
//                    case kAudioToolboxErr_InvalidSequenceType :
//                        AKLog( "Error: kAudioToolboxErr_InvalidSequenceType ")
//
//                    case kAudioToolboxErr_TrackIndexError :
//                        AKLog( "Error: kAudioToolboxErr_TrackIndexError ")
//
//                    case kAudioToolboxErr_TrackNotFound :
//                        AKLog( "Error: kAudioToolboxErr_TrackNotFound ")
//
//                    case kAudioToolboxErr_EndOfTrack :
//                        AKLog( "Error: kAudioToolboxErr_EndOfTrack ")
//
//                    case kAudioToolboxErr_StartOfTrack :
//                        AKLog( "Error: kAudioToolboxErr_StartOfTrack ")
//
//                    case kAudioToolboxErr_IllegalTrackDestination :
//                        AKLog( "Error: kAudioToolboxErr_IllegalTrackDestination")
//
//                    case kAudioToolboxErr_NoSequence :
//                        AKLog( "Error: kAudioToolboxErr_NoSequence ")
//
//                    case kAudioToolboxErr_InvalidEventType :
//                        AKLog( "Error: kAudioToolboxErr_InvalidEventType")
//
//                    case kAudioToolboxErr_InvalidPlayerState :
//                        AKLog( "Error: kAudioToolboxErr_InvalidPlayerState")
//
//                    case kAudioUnitErr_InvalidProperty :
//                        AKLog( "Error: kAudioUnitErr_InvalidProperty")
//
//                    case kAudioUnitErr_InvalidParameter :
//                        AKLog( "Error: kAudioUnitErr_InvalidParameter")
//
//                    case kAudioUnitErr_InvalidElement :
//                        AKLog( "Error: kAudioUnitErr_InvalidElement")
//
//                    case kAudioUnitErr_NoConnection :
//                        AKLog( "Error: kAudioUnitErr_NoConnection")
//
//                    case kAudioUnitErr_FailedInitialization :
//                        AKLog( "Error: kAudioUnitErr_FailedInitialization")
//
//                    case kAudioUnitErr_TooManyFramesToProcess :
//                        AKLog( "Error: kAudioUnitErr_TooManyFramesToProcess")
//
//                    case kAudioUnitErr_InvalidFile :
//                        AKLog( "Error: kAudioUnitErr_InvalidFile")
//
//                    case kAudioUnitErr_FormatNotSupported :
//                        AKLog( "Error: kAudioUnitErr_FormatNotSupported")
//
//                    case kAudioUnitErr_Uninitialized :
//                        AKLog( "Error: kAudioUnitErr_Uninitialized")
//
//                    case kAudioUnitErr_InvalidScope :
//                        AKLog( "Error: kAudioUnitErr_InvalidScope")
//
//                    case kAudioUnitErr_PropertyNotWritable :
//                        AKLog( "Error: kAudioUnitErr_PropertyNotWritable")
//
//                    case kAudioUnitErr_InvalidPropertyValue :
//                        AKLog( "Error: kAudioUnitErr_InvalidPropertyValue")
//
//                    case kAudioUnitErr_PropertyNotInUse :
//                        AKLog( "Error: kAudioUnitErr_PropertyNotInUse")
//
//                    case kAudioUnitErr_Initialized :
//                        AKLog( "Error: kAudioUnitErr_Initialized")
//
//                    case kAudioUnitErr_InvalidOfflineRender :
//                        AKLog( "Error: kAudioUnitErr_InvalidOfflineRender")
//
//                    case kAudioUnitErr_Unauthorized :
//                        AKLog( "Error: kAudioUnitErr_Unauthorized")
//
//                    default:
//                        print("Error: \(error)")
//                    }
//                #endif
        print(error)
    }
}

