import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:http/http.dart' as http;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';

// ===== دالة عليا للتشغيل في Isolate =====
Future<OrtSession?> _initializeVadIsolate(Uint8List modelBytes) async {
  try {
    final options = OrtSessionOptions();
    final session = await OrtSession.fromBuffer(modelBytes, options);
    debugPrint("🧠 VAD Initialized in Isolate Successfully");
    return session;
  } catch (e) {
    debugPrint("❌ Failed to initialize VAD in Isolate: $e");
    return null;
  }
}

// ===== خدمة VAD =====
class VadService {
  static const int sampleRate = 16000;
  static const int chunkSize = 1536;
  OrtSession? _session;
  OrtValue? _h, _c;
  bool _isSpeaking = false;
  final List<int> _audioBuffer = [];

  bool setupSession(OrtSession? session) {
    if (session == null) return false;
    _session = session;
    final hShape = [2, 1, 64];
    _h = OrtValueTensor.createTensorWithDataList(
        Float32List(2 * 1 * 64), hShape);
    final cShape = [2, 1, 64];
    _c = OrtValueTensor.createTensorWithDataList(
        Float32List(2 * 1 * 64), cShape);
    return true;
  }

  Future<void> processAudioChunk(
      {required Uint8List rawPcm16,
      required VoidCallback onSpeechStart,
      required VoidCallback onSpeechEnd}) async {
    if (_session == null) return;
    _audioBuffer.addAll(rawPcm16);
    while (_audioBuffer.length >= chunkSize * 2) {
      final chunkBytes = _audioBuffer.sublist(0, chunkSize * 2);
      _audioBuffer.removeRange(0, chunkSize * 2);
      final floatChunk = _convertBytesToFloat32(Uint8List.fromList(chunkBytes));
      final inputOrt =
          OrtValueTensor.createTensorWithDataList(floatChunk, [1, chunkSize]);
      final srOrt = OrtValueTensor.createTensorWithDataList(
          Int64List.fromList([sampleRate]), [1]);
      final inputs = {'input': inputOrt, 'sr': srOrt, 'h': _h!, 'c': _c!};
      final runOptions = OrtRunOptions();
      List<OrtValue?>? outputs;
      try {
        outputs = await _session!.runAsync(runOptions, inputs);
        if (outputs != null && outputs.isNotEmpty && outputs[0] != null) {
          final speechProbabilityValue =
              outputs[0]!.value as List<List<double>>;
          final speechProbability = speechProbabilityValue[0][0];
          final isSpeech = speechProbability > 0.5;
          if (isSpeech && !_isSpeaking) {
            _isSpeaking = true;
            onSpeechStart();
          } else if (!isSpeech && _isSpeaking) {
            _isSpeaking = false;
            onSpeechEnd();
          }
          if (outputs.length >= 3) {
            final newH = outputs[1];
            final newC = outputs[2];
            _h?.release();
            _c?.release();
            _h = newH;
            _c = newC;
          }
        }
      } catch (e) {
        debugPrint("Error running VAD model: $e");
      } finally {
        inputOrt.release();
        srOrt.release();
        runOptions.release();
        outputs?.forEach((val) => val?.release());
      }
    }
  }

  Float32List _convertBytesToFloat32(Uint8List bytes) {
    final byteData = ByteData.sublistView(bytes);
    final float32List = Float32List(chunkSize);
    for (int i = 0; i < chunkSize; i++) {
      final int16Sample = byteData.getInt16(i * 2, Endian.little);
      float32List[i] = int16Sample / 32768.0;
    }
    return float32List;
  }

  void dispose() {
    _h?.release();
    _c?.release();
    _session?.release();
    debugPrint("🧠 Silero VAD Disposed");
  }
}

class DeafPage extends StatefulWidget {
  const DeafPage({super.key});
  @override
  State<DeafPage> createState() => _DeafPageState();
}

class _DeafPageState extends State<DeafPage>
    with WidgetsBindingObserver, TickerProviderStateMixin {
  final TextEditingController _textController = TextEditingController();
  final TextEditingController _apiController = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  late FlutterSoundRecorder _recorder;
  late VadService _vadService;
  String _apiUrl = '';
  Directory? _tempDir;
  String? _currentFilePath;
  bool _microphonePermissionGranted = false;
  bool _isVadInitialized = false;
  bool _isListening = false;
  bool _isRecordingToFile = false;
  bool _isProcessing = false;
  bool _inSpeechSegment = false;
  StreamController<Uint8List>? _audioStreamController;
  StreamSubscription? _audioStreamSubscription;
  late AnimationController _micAnimationController;
  late AnimationController _pulseAnimationController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    _vadService = VadService();
    _recorder = FlutterSoundRecorder();
    WidgetsBinding.instance.addObserver(this);
    _setupAnimations();
    _initialize();
  }

  Future<void> _initialize() async {
    await _checkPermissions();
    if (_microphonePermissionGranted) {
      await _recorder.openRecorder();
    }
    _tempDir = await getTemporaryDirectory();
    await Future.wait([
      _loadSettings(),
      _initializeVadModel(),
    ]);
  }

  Future<void> _initializeVadModel() async {
    try {
      final modelData = await rootBundle.load('assets/silero_vad.onnx');
      // IMPORTANT: OrtSession (native object) cannot be sent across isolates.
      // Create the OrtSession here in the main isolate.
      final options = OrtSessionOptions();
      final session =
          await OrtSession.fromBuffer(modelData.buffer.asUint8List(), options);
      final success = _vadService.setupSession(session);
      debugPrint("🧠 VAD session created on main isolate: $success");
      if (mounted) setState(() => _isVadInitialized = success);
      if (!success) _showErrorSnackbar('فشل تهيئة نموذج VAD');
    } catch (e, st) {
      debugPrint('❌ Error initializing VAD on main isolate: $e\n$st');
      if (mounted) _showErrorSnackbar('خطأ في تهيئة نموذج VAD: $e');
    }
  }

  void _setupAnimations() {
    _micAnimationController = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 300));
    _pulseAnimationController = AnimationController(
        vsync: this, duration: const Duration(milliseconds: 1500));
    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.2).animate(
        CurvedAnimation(
            parent: _pulseAnimationController, curve: Curves.easeInOut));
  }

  void _onSpeechStart() {
    if (!_isListening || !mounted) return;
    debugPrint("🎤 Speech Started (AI VAD)");
    setState(() => _inSpeechSegment = true);
    if (!_isRecordingToFile) _startRecordingSegment();
  }

  void _onSpeechEnd() {
    if (!_isListening || !mounted) return;
    debugPrint("🤫 Speech Ended (AI VAD)");
    setState(() => _inSpeechSegment = false);
    if (_isRecordingToFile) _stopAndProcessSegment();
  }

  Future<void> _toggleVADListening() async {
    if (!_isVadInitialized) {
      _showErrorSnackbar('يرجى الانتظار، جاري تهيئة النموذج...');
      return;
    }
    if (!_microphonePermissionGranted) {
      _showPermissionDialog();
      await _checkPermissions();
      return;
    }
    if (_apiUrl.isEmpty) {
      _showErrorSnackbar('يرجى تعيين رابط API');
      return;
    }

    setState(() {
      if (_isListening) {
        _stopVADListening();
      } else {
        _startVADListening();
      }
    });
  }

  Future<void> _startVADListening() async {
    _isListening = true;
    _micAnimationController.forward();
    _pulseAnimationController.repeat(reverse: true);
    _audioStreamController = StreamController<Uint8List>();
    _audioStreamSubscription =
        _audioStreamController!.stream.listen((audioChunk) {
      _vadService.processAudioChunk(
          rawPcm16: audioChunk,
          onSpeechStart: _onSpeechStart,
          onSpeechEnd: _onSpeechEnd);
    });

    // --- السطر المسبب للمشكلة تم إزالته ---
    // لم نعد بحاجة للتحقق من `isOpen` لأننا فتحناه في `_initialize`

    await _recorder.startRecorder(
        toStream: _audioStreamController!.sink,
        codec: Codec.pcm16,
        numChannels: 1,
        sampleRate: VadService.sampleRate);
    _showSuccessSnackbar('بدء الاستماع... في انتظار الكلام');
  }

  Future<void> _stopVADListening() async {
    _isListening = false;
    if (_recorder.isRecording) await _recorder.stopRecorder();
    await _audioStreamSubscription?.cancel();
    await _audioStreamController?.close();
    if (_isRecordingToFile)
      await _stopAndProcessSegment(fromStopListening: true);
    _micAnimationController.reverse();
    _pulseAnimationController.stop();
    _pulseAnimationController.reset();
    _showSuccessSnackbar('تم إيقاف الاستماع');
  }

  Future<void> _startRecordingSegment() async {
    try {
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      _currentFilePath = '${_tempDir!.path}/vad_segment_$timestamp.m4a';
      await _recorder.startRecorder(
          toFile: _currentFilePath, codec: Codec.aacMP4);
      _isRecordingToFile = true;
      debugPrint('📹 بدء تسجيل مقطع جديد: $_currentFilePath');
    } catch (e) {
      debugPrint('Start recording segment error: $e');
    }
  }

  Future<void> _stopAndProcessSegment({bool fromStopListening = false}) async {
    try {
      String? path = _currentFilePath;
      if (!fromStopListening && _recorder.isRecording) {
        path = await _recorder.stopRecorder();
      }
      _isRecordingToFile = false;
      if (path != null && await File(path).exists()) {
        unawaited(_processAudioFile(File(path)));
      }
    } catch (e) {
      debugPrint('Stop and process segment error: $e');
    }
  }

  Future<void> _processAudioFile(File audioFile) async {
    if (!mounted) return;
    setState(() => _isProcessing = true);
    try {
      final uri = Uri.parse('$_apiUrl/stt');
      final request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('file', audioFile.path));
      final response =
          await request.send().timeout(const Duration(seconds: 45));
      final responseData = await response.stream.bytesToString();
      if (response.statusCode == 200) {
        final json = jsonDecode(responseData);
        final text = (json['text'] as String?)?.trim() ?? '';
        if (text.isNotEmpty && mounted) {
          setState(() {
            if (_textController.text.isNotEmpty) _textController.text += '\n\n';
            _textController.text += text;
          });
          WidgetsBinding.instance.addPostFrameCallback((_) {
            if (_scrollController.hasClients) {
              _scrollController.animateTo(
                  _scrollController.position.maxScrollExtent,
                  duration: const Duration(milliseconds: 300),
                  curve: Curves.easeOut);
            }
          });
        }
      } else {
        _showErrorSnackbar('خطأ في الخادم: ${response.statusCode}');
      }
    } catch (e) {
      _showErrorSnackbar('خطأ في الرفع');
    } finally {
      if (mounted) setState(() => _isProcessing = false);
      try {
        if (audioFile.path.contains('vad_segment')) {
          if (await audioFile.exists()) await audioFile.delete();
        }
      } catch (e) {
        debugPrint("Failed to delete temp file: $e");
      }
    }
  }

  Future<void> _pickAndProcessFile() async {
    try {
      final result = await FilePicker.platform.pickFiles(type: FileType.audio);
      if (result != null && result.files.single.path != null) {
        final audioFile = File(result.files.single.path!);
        _showSuccessSnackbar('تم اختيار الملف، جاري المعالجة...');
        unawaited(_processAudioFile(audioFile));
      }
    } catch (e) {
      _showErrorSnackbar('فشل اختيار الملف: $e');
    }
  }

  @override
  void dispose() {
    _stopVADListening();
    _vadService.dispose();
    _recorder.closeRecorder();
    _micAnimationController.dispose();
    _pulseAnimationController.dispose();
    _textController.dispose();
    _apiController.dispose();
    _scrollController.dispose();
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused ||
        state == AppLifecycleState.detached) {
      if (_isListening) _stopVADListening();
    }
  }

  @override
  Widget build(BuildContext context) {
    final micColor = !_microphonePermissionGranted
        ? Colors.grey
        : (_isListening
            ? (_inSpeechSegment ? Colors.green : Colors.orange)
            : Colors.blue);
    return Scaffold(
      appBar: AppBar(
        title: const Text('المحاضر الذكي (AI VAD)'),
        actions: [
          IconButton(
            icon: const Icon(Icons.upload_file),
            onPressed:
                (_isListening || _isProcessing) ? null : _pickAndProcessFile,
            tooltip: 'رفع ملف صوتي',
          ),
          IconButton(
              icon: const Icon(Icons.settings),
              onPressed: _showSettingsDialog,
              tooltip: 'الإعدادات'),
          IconButton(
              icon: const Icon(Icons.clear_all),
              onPressed: _clearText,
              tooltip: 'مسح النص'),
        ],
      ),
      body: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            margin: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: micColor.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: micColor.withOpacity(0.3)),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                if (!_isVadInitialized)
                  const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2))
                else
                  Icon(Icons.psychology, color: micColor),
                const SizedBox(width: 12),
                Text(_getStatusText(),
                    style: TextStyle(
                        color: micColor,
                        fontWeight: FontWeight.bold,
                        fontSize: 16)),
                if (_isProcessing) ...[
                  const Spacer(),
                  const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2)),
                ]
              ],
            ),
          ),
          Expanded(
            child: Container(
              margin: const EdgeInsets.fromLTRB(12, 0, 12, 12),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(color: Colors.grey.withOpacity(0.1), blurRadius: 10)
                ],
              ),
              child: TextField(
                controller: _textController,
                scrollController: _scrollController,
                maxLines: null,
                expands: true,
                textAlign: TextAlign.right,
                textDirection: TextDirection.rtl,
                style: const TextStyle(fontSize: 16, height: 1.6),
                decoration: InputDecoration(
                    hintText: 'سيظهر النص المحول هنا...',
                    border: InputBorder.none),
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(20),
            child: GestureDetector(
              onTap: _toggleVADListening,
              child: AnimatedBuilder(
                animation: _pulseAnimation,
                builder: (context, child) {
                  final scale = _isListening ? _pulseAnimation.value : 1.0;
                  return Transform.scale(
                    scale: scale,
                    child: Container(
                      width: 100,
                      height: 100,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: _isVadInitialized ? micColor : Colors.grey,
                        boxShadow: [
                          BoxShadow(
                              color: micColor.withOpacity(0.4),
                              blurRadius: 15,
                              spreadRadius: 5)
                        ],
                      ),
                      child: Center(
                        child: _isProcessing
                            ? const CircularProgressIndicator(
                                color: Colors.white)
                            : Icon(_isListening ? Icons.stop : Icons.mic,
                                color: Colors.white, size: 65),
                      ),
                    ),
                  );
                },
              ),
            ),
          ),
        ],
      ),
    );
  }

  String _getStatusText() {
    if (!_isVadInitialized) return 'جاري تهيئة نموذج الذكاء الاصطناعي...';
    if (_isListening)
      return _inSpeechSegment ? 'يتم اكتشاف الكلام...' : 'في انتظار الكلام...';
    return 'اضغط على الميكروفون للبدء';
  }

  void _clearText() {
    if (mounted) setState(() => _textController.clear());
  }

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    if (mounted) {
      setState(() {
        _apiUrl = prefs.getString('whisper_api_url') ?? '';
        _apiController.text = _apiUrl;
      });
    }
  }

  Future<void> _saveSettings() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('whisper_api_url', _apiUrl);
  }

  Future<void> _checkPermissions() async {
    final status = await Permission.microphone.request();
    if (mounted)
      setState(() =>
          _microphonePermissionGranted = status == PermissionStatus.granted);
  }

  void _showSettingsDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('الإعدادات'),
        content: TextField(
          controller: _apiController,
          decoration: const InputDecoration(
              border: OutlineInputBorder(), labelText: 'رابط API'),
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('إلغاء')),
          ElevatedButton(
            onPressed: () {
              _apiUrl =
                  _apiController.text.trim().replaceAll(RegExp(r'/$'), '');
              _saveSettings();
              Navigator.pop(context);
              _showSuccessSnackbar('تم حفظ الإعدادات');
            },
            child: const Text('حفظ'),
          ),
        ],
      ),
    );
  }

  void _showPermissionDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('إذن الميكروفون مطلوب'),
        content: const Text(
            'يحتاج التطبيق إذن الميكروفون للاستماع وتسجيل المقاطع الصوتية.'),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('حسنًا')),
          TextButton(
              onPressed: () async {
                Navigator.pop(context);
                await openAppSettings();
              },
              child: const Text('فتح الإعدادات')),
        ],
      ),
    );
  }

  void _showErrorSnackbar(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(SnackBar(content: Text(msg), backgroundColor: Colors.red));
  }

  void _showSuccessSnackbar(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
          SnackBar(content: Text(msg), backgroundColor: Colors.green));
  }
}

void unawaited(Future<void> future) {}
