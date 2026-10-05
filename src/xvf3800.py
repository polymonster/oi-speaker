"""USB control for the reSpeaker XVF3800 4-Mic Array (LED ring, DoA, tuning params).

Vendored and tidied from https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY
(`python_control/xvf_host.py`) — the device speaks a vendor control protocol over USB,
so this is a pure pyusb implementation of their `xvf_host` binary.

Library use:

    from xvf3800 import XVF3800, Effect

    ring = XVF3800.open()
    if ring:
        ring.solid(0xFF8800)          # single colour
        ring.breathe(0x0000FF, 1)     # breathing at speed 1
        ring.ring([0xFF0000] * 12)    # per-LED colours
        ring.off()

CLI use (same surface as upstream):

    python src/xvf3800.py --list
    python src/xvf3800.py VERSION
    python src/xvf3800.py LED_COLOR --values 0xFF0000
"""

import sys
import struct
import argparse
import time

import usb.core
import usb.util

CONTROL_SUCCESS = 0
SERVICER_COMMAND_RETRY = 64

VENDOR_ID = 0x2886
PRODUCT_ID = 0x001A

LED_COUNT = 12

# name: (resid, cmdid, length, access, type, description)
PARAMETERS = {
    # APPLICATION_SERVICER_RESID commands
    "VERSION": (48, 0, 3, "ro", "uint8", "The version number of the firmware. The format is VERSION_MAJOR VERSION_MINOR VERSION_PATCH"),
    "BLD_MSG": (48, 1, 50, "ro", "char", "Retrieve the build message built in to the firmware, normally the build configuration name"),
    "BLD_HOST": (48, 2, 30, "ro", "char", "Retrieve details of the CI build host used to build the firmware"),
    "BLD_REPO_HASH": (48, 3, 40, "ro", "char", "Retrieve the GIT hash of the sw_xvf3800 repo used to build the firmware"),
    "BLD_MODIFIED": (48, 4, 6, "ro", "char", "Show whether or not the current firmware repo has been modified from the official release. Requires use of a GIT repo"),
    "BOOT_STATUS": (48, 5, 3, "ro", "char", "Shows whether or not the firmware has been booted via SPI or JTAG/FLASH"),
    "TEST_CORE_BURN": (48, 6, 1, "rw", "uint8", "Set to enable core burn to exercise worst case timing. This will reboot the chip, reset all parameters to default and significantly increase power consumption."),
    "REBOOT": (48, 7, 1, "wo", "uint8", "Set to any value to reboot the chip and reset all parameters to default."),
    "USB_BIT_DEPTH": (48, 8, 2, "rw", "uint8", "Only relevant for the UA device variant. For the UA device, set or get the USB bit depth IN, OUT to either 16, 24 or 32. Setting will reboot the chip, resetting all other parameters to default. If issued to the INT device, a set is ignored and the device is not rebooted while a get always returns 0 as both IN and OUT bit depths."),
    "SAVE_CONFIGURATION": (48, 9, 1, "wo", "uint8", "Set to any value to save the current configuration to flash."),
    "CLEAR_CONFIGURATION": (48, 10, 1, "wo", "uint8", "Set to any value to clear the current configuration and revert to the default configuration."),
    "AIC3104_HP_LEVEL": (48, 11, 1, "rw", "uint8", "Set or get the AIC3104 headphone level. Valid range: [0 .. 9]"),
    "AIC3104_LINEOUT_LEVEL": (48, 12, 1, "rw", "uint8", "Set or get the AIC3104 lineout level. Valid range: [0 .. 9]"),

    # AEC_RESID commands
    "SHF_BYPASS": (33, 70, 1, "rw", "uint8", "AEC bypass"),
    "AEC_NUM_MICS": (33, 71, 1, "ro", "int32", "Number of microphone inputs into the AEC"),
    "AEC_NUM_FARENDS": (33, 72, 1, "ro", "int32", "Number of farend inputs into the AEC"),
    "AEC_MIC_ARRAY_TYPE": (33, 73, 1, "ro", "int32", "Microphone array type (1 - linear, 2 - squarecular)"),
    "AEC_MIC_ARRAY_GEO": (33, 74, 12, "ro", "float", "Microphone array geometry. Each microphone is represented by 3 XYZ coordinates in m."),
    "AEC_AZIMUTH_VALUES": (33, 75, 4, "ro", "radians", "Azimuth values in radians - beam 1, beam 2, free-running beam, 3 - auto-select beam"),
    "TEST_AEC_DISABLE_CONTROL": (33, 76, 1, "wo", "uint32", "Disable Control processing in the AEC resource. Only to be used for internal testing! Do not use in normal operation. When disabled, there's no way to bring back control without restarting the device."),
    "AEC_CURRENT_IDLE_TIME": (33, 77, 1, "ro", "uint32", "AEC processing current idle time in 10ns ticks"),
    "AEC_MIN_IDLE_TIME": (33, 78, 1, "ro", "uint32", "AEC processing minimum idle time in 10ns ticks"),
    "AEC_RESET_MIN_IDLE_TIME": (33, 79, 1, "wo", "uint32", "Reset the AEC minimum idle time. The value of the argument passed is ignored."),
    "AEC_SPENERGY_VALUES": (33, 80, 4, "ro", "float", "Speech energy level values for each beam. Any value above 0 indicates speech. Higher values indicates louder or closer speech, however noise, echo and reverb can cause the energy level to decrease. 0 - beam 1, 1 - beam 2, 2 - free-running beam, 3 - auto-select beam"),
    "AEC_FIXEDBEAMSAZIMUTH_VALUES": (33, 81, 2, "rw", "radians", "Azimuth values in radians for beams in fixed mode - fixed beam 1, fixed beam 2"),
    "AEC_FIXEDBEAMSELEVATION_VALUES": (33, 82, 2, "rw", "radians", "Elevation angles in radians for the beams in fixed mode. 0 - fixed beam 1, 1 - fixed beam 2"),
    "AEC_FIXEDBEAMSGATING": (33, 83, 1, "rw", "uint8", "Enables/disables gating for beams in fixed mode. When enabled, speech energy is used to determine which beam should be active; inactive beams are silenced."),
    "SPECIAL_CMD_AEC_FAR_MIC_INDEX": (33, 90, 2, "wo", "int32", "Set Far and Mic index for AEC filter read. Also used as a trigger command that starts a new filter read"),
    "SPECIAL_CMD_AEC_FILTER_COEFF_START_OFFSET": (33, 91, 1, "rw", "int32", "start offset to get or set the filter coefficients from"),
    "SPECIAL_CMD_AEC_FILTER_COEFFS": (33, 92, 15, "rw", "float", "get or set AEC filter coefficients"),
    "SPECIAL_CMD_AEC_FILTER_LENGTH": (33, 93, 1, "ro", "int32", "get AEC filter length"),
    "AEC_FILTER_CMD_ABORT": (33, 94, 1, "wo", "int32", "Reset the special command state machine. Used for safely exiting from an AEC filter read/write command sequence that has gone wrong."),
    "AEC_AECPATHCHANGE": (33, 0, 1, "ro", "int32", "AEC Path Change Detection. Valid range: 0,1 (false,true)"),
    "AEC_HPFONOFF": (33, 1, 1, "rw", "int32", "High-pass Filter on microphone signals. Valid range: 0,1,2,3,4 (0:Off, 1:on70, 2:on125, 3:on150, 4:on180)"),
    "AEC_AECSILENCELEVEL": (33, 2, 2, "rw", "float", "Power threshold for signal detection in adaptive filter.(set,cur),  Valid range (set): [0.0 .. 1.0] Valid range (cur): 0.05*set, 1e-6f or set."),
    "AEC_AECCONVERGED": (33, 3, 1, "ro", "int32", "Flag indicating whether AEC is converged. Valid range: 0,1 (false,true)"),
    "AEC_AECEMPHASISONOFF": (33, 4, 1, "rw", "int32", "Pre-emphasis and de-emphasis filtering for AEC. Valid range: 0,1,2 (off,on,on_eq)on: Emphasis filter for speech signals without modification of far-end ref. on_eq: Emphasis filter for far-end reference signals where the low frequencies are boosted by e.g. an equalizer."),
    "AEC_FAR_EXTGAIN": (33, 5, 1, "rw", "float", "External gain in dB applied to the far-end reference signals. Valid range: [-inf .. inf]"),
    "AEC_PCD_COUPLINGI": (33, 6, 1, "rw", "float", "Sensitivity parameter for PCD. Valid range: [0.0 .. 1.0]PCD can be disabled by setting a value outside the range"),
    "AEC_PCD_MINTHR": (33, 7, 1, "rw", "float", "Minimum threshold value used in PCD. Valid range: [0.0 .. 0.02]"),
    "AEC_PCD_MAXTHR": (33, 8, 1, "rw", "float", "Maximum threshold value used in PCD. Valid range: [0.025 .. 0.2]"),
    "AEC_RT60": (33, 9, 1, "ro", "float", "Current RT60 estimate. Valid range: [0.250 .. 0.900] (seconds) A negative value indicates that the RT60 estimation is invalid."),
    "AEC_ASROUTONOFF": (33, 35, 1, "rw", "int32", "Enables the automatic speech recognition (ASR) output: if set to 0, the AEC residuals are output, one channel per microphone, if set to 1, the ASR processed output is used, where each channel is associated with a beam from the beamformer. Valid range: 0,1 (off, on)"),
    "AEC_ASROUTGAIN": (33, 36, 1, "rw", "float", "Fixed gain applied to qsr output, i.e., only applied if ASROUTONOFF==1.. Valid range: [0.0 .. 1000.0] (linear gain factor)"),
    "AEC_FIXEDBEAMSONOFF": (33, 37, 1, "rw", "int32", "Enables or disables fixed focused beam mode. Valid range: 0,1 (off, on)"),
    "AEC_FIXEDBEAMNOISETHR": (33, 38, 2, "rw", "float", "Threshold value for updating the noise canceller when fixed beam mode is enabled. A higher value indicates that the noise canceller may update when the free running beam is close to the fixed beam. A lower value indicates that the noise canceller may update when the free running beam is further away from the focused beam.[BECLEAR_NUMBER_OF_BEAMS],  Valid range: [0.0 .. 1.0]"),

    # AUDIO_MGR_RESID commands
    "AUDIO_MGR_MIC_GAIN": (35, 0, 1, "rw", "float", "Audio Mgr pre SHF microphone gain"),
    "AUDIO_MGR_REF_GAIN": (35, 1, 1, "rw", "float", "Audio Mgr pre SHF reference gain"),
    "AUDIO_MGR_CURRENT_IDLE_TIME": (35, 2, 1, "ro", "int32", "Get audio manager current idle time in 10ns ticks"),
    "AUDIO_MGR_MIN_IDLE_TIME": (35, 3, 1, "ro", "int32", "Get audio manager min idle time in 10ns ticks"),
    "AUDIO_MGR_RESET_MIN_IDLE_TIME": (35, 4, 1, "wo", "int32", "Reset audio manager min idle time. The value of the argument passed is ignored."),
    "MAX_CONTROL_TIME": (35, 5, 1, "ro", "int32", "Get audio manager max control time"),
    "RESET_MAX_CONTROL_TIME": (35, 6, 1, "wo", "int32", "Reset audio manager max control time. The value of the argument passed is ignored."),
    "I2S_CURRENT_IDLE_TIME": (35, 7, 1, "ro", "int32", "Get I2S current idle time in 10ns ticks"),
    "I2S_MIN_IDLE_TIME": (35, 8, 1, "ro", "int32", "Get I2S min idle time in 10ns ticks"),
    "I2S_RESET_MIN_IDLE_TIME": (35, 9, 1, "wo", "int32", "I2S reset idle time. The value of the argument passed is ignored."),
    "I2S_INPUT_PACKED": (35, 10, 1, "rw", "uint8", "Will expect packed input on both I2S or USB channels if this is not 0. Note, could take up to 3 samples to take effect."),
    "AUDIO_MGR_SELECTED_AZIMUTHS": (35, 11, 2, "ro", "radians", "This returns the azimuths determined by the beam_selection function in post_shf_dsp.c. The default implementation will provide a processed DoA at index 0 and the DoA of the auto select beam at index 1. The processed DoA uses speech energy to select from the DoA on each fixed beam to provide the direction of a speaker. NAN will be returned if neither of the fixed beams contain speech."),
    "AUDIO_MGR_SELECTED_CHANNELS": (35, 12, 2, "rw", "uint8", "Default implementation of post processing will use this to select which channels should be output to MUX_USER_CHOSEN_CHANNELS. Note that a customer implementation of the beam selection stage could override this command. How this channel selection aligns with actual output depends on the mux configuration"),
    "AUDIO_MGR_OP_PACKED": (35, 13, 2, "rw", "uint8", "<L>, <R>; Sets/gets packing status for L and R output channels"),
    "AUDIO_MGR_OP_UPSAMPLE": (35, 14, 2, "rw", "uint8", "<L>, <R>; Sets/gets upsample status for L and R output channels, where appropriate"),
    "AUDIO_MGR_OP_L": (35, 15, 2, "rw", "uint8", "<category>, <source>; Sets category and source for L output channel. Equivalent to AUDIO_MGR_OP_L_PK0"),
    "AUDIO_MGR_OP_L_PK0": (35, 16, 2, "rw", "uint8", "<category>, <source>; Sets category and source for first (of three) sources on the L channel in packed mode. Equivalent to AUDIO_MGR_OP_L"),
    "AUDIO_MGR_OP_L_PK1": (35, 17, 2, "rw", "uint8", "Sets category and source for second (of three) sources on the L channel in packed mode"),
    "AUDIO_MGR_OP_L_PK2": (35, 18, 2, "rw", "uint8", "Sets category and source for third (of three) sources on the L channel in packed mode"),
    "AUDIO_MGR_OP_R": (35, 19, 2, "rw", "uint8", "<category>, <source>; Sets category and source for R output channel. Equivalent to AUDIO_MGR_OP_R_PK0"),
    "AUDIO_MGR_OP_R_PK0": (35, 20, 2, "rw", "uint8", "<category>, <source>; Sets category and source for first (of three) sources on the R channel in packed mode. Equivalent to AUDIO_MGR_OP_R"),
    "AUDIO_MGR_OP_R_PK1": (35, 21, 2, "rw", "uint8", "Sets category and source for second (of three) sources on the R channel in packed mode"),
    "AUDIO_MGR_OP_R_PK2": (35, 22, 2, "rw", "uint8", "Sets category and source for third (of three) sources on the R channel in packed mode"),
    "AUDIO_MGR_OP_ALL": (35, 23, 12, "rw", "uint8", "Sets category and source for all 3 sources on L channel and all 3 sources for R channel. Equivalent to AUDIO_MGR_OP_[L,R]_PK[0,1,2] with successive pairs of arguments"),
    "I2S_INACTIVE": (35, 24, 1, "ro", "uint8", "Returns whether the main audio loop is exchanging samples with I2S (0). If not (1), I2S is inactive"),
    "AUDIO_MGR_FAR_END_DSP_ENABLE": (35, 25, 1, "rw", "uint8", "Enables/disables XVF3800 far-end DSP (if implemented). Write a 1 to enable, 0 to disable"),
    "AUDIO_MGR_SYS_DELAY": (35, 26, 1, "rw", "int32", "Delay, measured in samples, that is applied to the reference signal before passing to SHF algorithm"),
    "I2S_DAC_DSP_ENABLE": (35, 27, 1, "rw", "uint8", "Indicates if the DAC performs DSP on the far-end reference signal. If enabled (1), use the I2S signal as input to the audio pipeline, even in the UA configuration"),

    # GPO_SERVICER_RESID commands — the LED ring lives here
    "GPO_READ_VALUES": (20, 0, 5, "ro", "uint8", "Get current logic level of all GPO pins, in order of Pin X0D11, X0D30, X0D31, X0D33 and X0D39."),
    "GPO_WRITE_VALUE": (20, 1, 2, "wo", "uint8", "Set current logic level of selected GPO pin. Supports Pin X0D11, X0D30, X0D31, X0D33 and X0D39."),
    "GPO_PORT_PIN_INDEX": (20, 2, 2, "rw", "uint32", "GPO port index and pin index that the following commands would be directed to"),
    "GPO_PIN_VAL": (20, 3, 3, "wo", "uint8", "value to write to one pin of a GPO port. Payload specifies port_index, pin_index and value to write to the pin"),
    "GPO_PIN_ACTIVE_LEVEL": (20, 4, 1, "rw", "uint32", "Active level of the port/pin specified by the GPO_PORT_PIN_INDEX command. 1 = ACTIVE_HIGH, 0 = ACTIVE_LOW"),
    "LED_EFFECT": (20, 12, 1, "rw", "uint8", "Set the LED effect mode, 0 = off, 1 = breath, 2 = rainbow, 3 = single color, 4 = doa, 5 = ring"),
    "LED_BRIGHTNESS": (20, 13, 1, "rw", "uint8", "Set the brightness value of the LED for the breath and rainbow mode"),
    "LED_GAMMIFY": (20, 14, 1, "rw", "uint8", "Set to enable gamma correction, 0 = disable, 1 = enable"),
    "LED_SPEED": (20, 15, 1, "rw", "uint8", "Set the effect speed of breath and rainbow mode"),
    "LED_COLOR": (20, 16, 1, "rw", "uint32", "Set the LED color of breath mode and single color mode"),
    "LED_DOA_COLOR": (20, 17, 2, "rw", "uint32", "Set the LED color of doa mode, the first value is the base color and the second value is the doa color"),
    "DOA_VALUE": (20, 18, 2, "ro", "uint16", "Get the doa value and if speech is detected, payload[0] = doa value, from 0 to 359, payload[1] = 1 if speech is detected, 0 if not"),
    "LED_RING_COLOR": (20, 19, 12, "rw", "uint32", "Set the LED color of ring mode, each value is the color of one LED"),

    # PP_RESID commands
    "PP_CURRENT_IDLE_TIME": (17, 70, 1, "ro", "uint32", "PP processing current idle time in 10ns ticks"),
    "PP_MIN_IDLE_TIME": (17, 71, 1, "ro", "uint32", "PP processing minimum idle time in 10ns ticks"),
    "PP_RESET_MIN_IDLE_TIME": (17, 72, 1, "wo", "uint32", "Reset the PP minimum idle time. The value of the argument passed is ignored."),
    "PP_NL_MODEL_CMD_ABORT": (17, 94, 1, "wo", "int32", "Reset the special command state machine. Used for safely exiting from a NL model read/write sequence that has gone wrong."),
    "PP_EQUALIZATION_CMD_ABORT": (17, 100, 1, "wo", "int32", "Reset the special command state machine. Used for safely exiting from an equalization read/write sequence that has gone wrong."),
    "PP_AGCONOFF": (17, 10, 1, "rw", "int32", "Automatic Gain Control. Valid range: 0,1 (off,on)"),
    "PP_AGCMAXGAIN": (17, 11, 1, "rw", "float", "Maximum AGC gain factor. Valid range: [1.0 .. 1000.0] (linear gain factor)"),
    "PP_AGCDESIREDLEVEL": (17, 12, 1, "rw", "float", "Target power level of the output signal. Valid range: [1e-8 .. 1.0] (power level)"),
    "PP_AGCGAIN": (17, 13, 1, "rw", "float", "Current AGC gain factor. Valid range: [1.0 .. 1000.0] (linear gain factor)"),
    "PP_AGCTIME": (17, 14, 1, "rw", "float", "Ramp-up/down time-constant. Valid range: [0.5 .. 4.0] (seconds)"),
    "PP_AGCFASTTIME": (17, 15, 1, "rw", "float", "Ramp down time-constant in case of peaks. Valid range: [0.05 .. 4.0] (seconds)"),
    "PP_AGCALPHAFASTGAIN": (17, 16, 1, "rw", "float", "Gain threshold enabling fast alpha mode. Valid range: [0.0 .. 1000.0] (linear gain factor)"),
    "PP_AGCALPHASLOW": (17, 17, 1, "rw", "float", "Slow memory parameter for speech power. Valid range [0.0 .. 1.0]"),
    "PP_AGCALPHAFAST": (17, 18, 1, "rw", "float", "Fast memory parameter for speech power. Valid range [0.0 .. 1.0]"),
    "PP_LIMITONOFF": (17, 19, 1, "rw", "int32", "Limiter on communication output. Valid range: 0,1 (off,on)"),
    "PP_LIMITPLIMIT": (17, 20, 1, "rw", "float", "Maximum limiter power. Valid range: [1e-8 .. 1.0] (power level)"),
    "PP_MIN_NS": (17, 21, 1, "rw", "float", "Gain-floor for stationary noise suppression. Valid range: [0.0 .. 1.0]"),
    "PP_MIN_NN": (17, 22, 1, "rw", "float", "Gain-floor for non-stationary noise suppression. Valid range: [0.0 .. 1.0]"),
    "PP_ECHOONOFF": (17, 23, 1, "rw", "int32", "Echo suppression. Valid range: 0,1 (off,on)"),
    "PP_GAMMA_E": (17, 24, 1, "rw", "float", "Over-subtraction factor of echo (direct and early components). Valid range: [0.0 .. 2.0]"),
    "PP_GAMMA_ETAIL": (17, 25, 1, "rw", "float", "Over-subtraction factor of echo (tail components). Valid range: [0.0 .. 2.0]"),
    "PP_GAMMA_ENL": (17, 26, 1, "rw", "float", "Over-subtraction factor of non-linear echo. Valid range: [0.0 .. 5.0]"),
    "PP_NLATTENONOFF": (17, 27, 1, "rw", "int32", "Non-Linear echo attenuation. Valid range: 0,1 (off,on)"),
    "PP_NLAEC_MODE": (17, 28, 1, "rw", "int32", "Non-Linear AEC training mode. Valid range: 0,1,2 (normal,train,train2)"),
    "PP_MGSCALE": (17, 29, 3, "rw", "float", "Minimum gain scale for acoustic echo suppression.(max,min,cur),  Valid range (max,min): [(1.0,0.0) .. (1e5,max)] Valid range (cur): min or max."),
    "PP_FMIN_SPEINDEX": (17, 30, 1, "rw", "float", "In case of double talk, frequencies below SPEINDEX are more suppressed than frequencies above SPEINDEX. The actual suppression is depending on the setting of DTSENSITIVE. Valid range: [0.0 .. 7999.0]"),
    "PP_DTSENSITIVE": (17, 31, 1, "rw", "int32", "Tradeoff between echo suppression and doubletalk performance. A lower value prefers high echo suppression, a higher value prefers better doubletalk performance. Valid range: [0 .. 5, 10 .. 15]"),
    "PP_ATTNS_MODE": (17, 32, 1, "rw", "int32", "Additional reduction of AGC gain during non-speech. Valid range: 0,1 (off, on)"),
    "PP_ATTNS_NOMINAL": (17, 33, 1, "rw", "float", "Amount of additional reduction during non-speech at nominal speech level. Valid range: [0.0 .. 1.0]"),
    "PP_ATTNS_SLOPE": (17, 34, 1, "rw", "float", "Determines the extra amount of suppression during non-speech when the AGC level increases at lower speech level. Valid range: [0.0 .. 5.0]"),
}


class Effect:
    """LED_EFFECT modes."""
    OFF = 0
    BREATH = 1
    RAINBOW = 2
    SOLID = 3
    DOA = 4
    RING = 5


class XVF3800:
    """Vendor control interface to the XVF3800 over USB."""

    TIMEOUT = 100000
    READ_RETRIES = 100

    def __init__(self, dev, debug: bool = False):
        self.dev = dev
        self.debug = debug

    @classmethod
    def open(cls, vid: int = VENDOR_ID, pid: int = PRODUCT_ID, debug: bool = False):
        """Find the device, or return None if it isn't plugged in / isn't accessible."""
        dev = None
        if sys.platform.startswith('win'):
            # windows has no system libusb — libusb-package ships one alongside the wheel
            import libusb_package
            dev = libusb_package.find(idVendor=vid, idProduct=pid)
        else:
            dev = usb.core.find(idVendor=vid, idProduct=pid)
        if not dev:
            return None
        return cls(dev, debug=debug)

    def write(self, name: str, values):
        data = PARAMETERS.get(name)
        if data is None:
            raise KeyError('unknown command {}'.format(name))

        resid, cmdid, data_cnt, access, data_type, _ = data
        if access == "ro":
            raise ValueError('{} is read-only'.format(name))

        if data_type == 'char':
            payload = bytearray(values, 'utf-8') if isinstance(values, str) else bytearray(values)
        else:
            if len(values) != data_cnt:
                raise ValueError('{} takes {} value(s), got {}'.format(name, data_cnt, len(values)))
            payload = []
            for i in range(data_cnt):
                if data_type == 'float' or data_type == 'radians':
                    payload += struct.pack(b'<f', float(values[i]))
                elif data_type == 'uint8':
                    payload += int(values[i]).to_bytes(1, byteorder='little')
                elif data_type == 'uint16':
                    payload += struct.pack(b'<H', int(values[i]))
                elif data_type == 'uint32':
                    payload += struct.pack(b'<I', int(values[i]))
                else:
                    payload += struct.pack(b'<i', int(values[i]))

        if self.debug:
            print("WriteCMD: cmdid: {}, resid: {}, payload: {}".format(cmdid, resid, bytes(payload)))

        self.dev.ctrl_transfer(
            usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, cmdid, resid, payload, self.TIMEOUT)

    def read(self, name: str):
        data = PARAMETERS.get(name)
        if data is None:
            raise KeyError('unknown command {}'.format(name))

        resid, cmdid, data_cnt, access, data_type, _ = data
        if access == "wo":
            raise ValueError('{} is write-only'.format(name))

        cmdid = 0x80 | cmdid # read bit
        if data_type == 'uint8' or data_type == 'char':
            width = 1
        elif data_type == 'uint16':
            width = 2
        else:
            width = 4
        length = data_cnt * width + 1 # 1 byte for status

        # the servicer answers RETRY while it is busy, so poll until it commits
        attempts = 0
        while True:
            response = self.dev.ctrl_transfer(
                usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0, cmdid, resid, length, self.TIMEOUT)
            if response[0] == CONTROL_SUCCESS:
                break
            if response[0] != SERVICER_COMMAND_RETRY:
                raise ValueError('unknown status code: {}'.format(response[0]))
            attempts += 1
            if attempts > self.READ_RETRIES:
                raise ValueError('read of {} retried more than {} times'.format(name, self.READ_RETRIES))
            time.sleep(0.01)

        if self.debug:
            print("ReadCMD: cmdid: {}, resid: {}, payload: {}".format(cmdid, resid, response.tolist()))

        payload = response.tobytes()[1:]
        if data_type == 'char':
            return payload.rstrip(b'\x00').decode('utf-8', errors='ignore')

        fmt = {'uint8': 'B', 'uint16': 'H', 'uint32': 'I', 'int32': 'i', 'float': 'f', 'radians': 'f'}[data_type]
        return struct.unpack('<' + fmt * data_cnt, payload)

    # -- ring helpers ------------------------------------------------------

    def set_effect(self, effect: int):
        self.write("LED_EFFECT", [effect])

    def set_color(self, rgb: int):
        self.write("LED_COLOR", [rgb])

    def set_brightness(self, brightness: int):
        """0..255, applies to breath and rainbow modes."""
        self.write("LED_BRIGHTNESS", [brightness])

    def set_speed(self, speed: int):
        """Effect speed for breath and rainbow modes."""
        self.write("LED_SPEED", [speed])

    def set_gamma(self, enable: bool):
        self.write("LED_GAMMIFY", [1 if enable else 0])

    def off(self):
        self.set_effect(Effect.OFF)

    def solid(self, rgb: int):
        """Light the whole ring one colour."""
        self.set_color(rgb)
        self.set_effect(Effect.SOLID)

    def breathe(self, rgb: int, speed: int = 1, brightness: int = 255):
        self.set_color(rgb)
        self.set_speed(speed)
        self.set_brightness(brightness)
        self.set_effect(Effect.BREATH)

    def rainbow(self, speed: int = 1, brightness: int = 255):
        self.set_speed(speed)
        self.set_brightness(brightness)
        self.set_effect(Effect.RAINBOW)

    def doa_mode(self, base_rgb: int = 0x000000, doa_rgb: int = 0x00FF00):
        """Firmware-driven direction-of-arrival indicator (the boot default)."""
        self.write("LED_DOA_COLOR", [base_rgb, doa_rgb])
        self.set_effect(Effect.DOA)

    def ring(self, colors):
        """Set all 12 LEDs individually. Shorter sequences are repeated to fill the ring."""
        colors = list(colors)
        if not colors:
            raise ValueError('ring() needs at least one colour')
        colors = [colors[i % len(colors)] for i in range(LED_COUNT)]
        self.write("LED_RING_COLOR", colors)
        self.set_effect(Effect.RING)

    def doa(self):
        """Returns (angle_degrees, speech_detected)."""
        angle, speech = self.read("DOA_VALUE")
        return angle, bool(speech)

    def version(self):
        return self.read("VERSION")

    def close(self):
        usb.util.dispose_resources(self.dev)


def find(vid: int = VENDOR_ID, pid: int = PRODUCT_ID, debug: bool = False):
    """Backwards-compatible alias for XVF3800.open()."""
    return XVF3800.open(vid=vid, pid=pid, debug=debug)


# -- cli -------------------------------------------------------------------

def parse_value(value_str: str):
    """Parse a value that may be decimal, or hex with a 0x or $ prefix."""
    try:
        if value_str.startswith(('0x', '0X')):
            return int(value_str, 16)
        elif value_str.startswith('$'):
            return int(value_str[1:], 16)
        try:
            return float(value_str)
        except ValueError:
            return int(value_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "cannot parse '{}' as a number (decimal, 0xFF or $FF)".format(value_str))


def case_insensitive_command(value: str):
    upper_value = value.upper()
    if upper_value not in PARAMETERS:
        similar = [cmd for cmd in PARAMETERS if upper_value in cmd]
        if similar:
            raise argparse.ArgumentTypeError(
                "invalid command '{}'. Did you mean one of: {}?".format(value, similar))
        raise argparse.ArgumentTypeError("invalid command '{}'".format(value))
    return upper_value


def list_commands():
    print("\n=== Supported Commands ===\n")
    print(f"{'Command':<45} {'RESID':<6} {'CMDID':<6} {'Length':<7} {'Access':<8} {'Type':<8} {'Description'}")
    print("-" * 120)
    for name, (resid, cmdid, length, access, param_type, description) in PARAMETERS.items():
        print(f"{name:<45} {resid:<6} {cmdid:<6} {length:<7} {access:<8} {param_type:<8} {description}")


def main():
    parser = argparse.ArgumentParser(description='reSpeaker XVF3800 host control')
    parser.add_argument('-l', '--list', action='store_true',
                        help='list all supported commands with detailed information')
    parser.add_argument('COMMAND', nargs='?', type=case_insensitive_command,
                        help='command to execute (e.g. VERSION, LED_COLOR, DOA_VALUE)')
    parser.add_argument('--vid', type=lambda x: int(x, 0), default=VENDOR_ID,
                        help='usb vendor ID (default: 0x2886)')
    parser.add_argument('--pid', type=lambda x: int(x, 0), default=PRODUCT_ID,
                        help='usb product ID (default: 0x001A)')
    parser.add_argument('--values', nargs='+', type=parse_value,
                        help='values for write commands, decimal (123, 1.5) or hex (0x7B, $7B)')
    parser.add_argument('--debug', action='store_true', help='print the raw usb transfers')

    args = parser.parse_args()

    if args.list:
        list_commands()
        sys.exit(0)

    if not args.COMMAND:
        parser.error('a COMMAND is required unless --list is given')

    dev = XVF3800.open(vid=args.vid, pid=args.pid, debug=args.debug)
    if not dev:
        print('No device found')
        sys.exit(1)

    try:
        if args.values:
            dev.write(args.COMMAND, args.values)
        else:
            result = dev.read(args.COMMAND)
            if args.COMMAND in ("LED_COLOR", "LED_DOA_COLOR", "LED_RING_COLOR"):
                formatted = [f"0x{val:06X}" for val in result]
            elif isinstance(result, str):
                formatted = [f"'{result}'"]
            else:
                formatted = [f"{val:.3f}" if isinstance(val, float) else str(val) for val in result]
            print(f"{args.COMMAND}: [{', '.join(formatted)}]")
        print("Done!")
    except Exception as e:
        print(f"Error executing command {args.COMMAND}: {e}")
        sys.exit(1)
    finally:
        dev.close()


if __name__ == '__main__':
    main()
