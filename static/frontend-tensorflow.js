// @ts-check
//Robotic Intuition Operator Tensorflow frontend
//Debashish Buragohain
"use-strict"
//test mode controller is in the frontend javascript file
let poseResponseAltSaved = [];
let CSE_apikey = "AIzaSyCgEEs4Mk8Db6WG1FeDQWYfmwsAKQGs1-Y";
let CSE_id = "6463da411225d7f00"
var webcamElement1 = document.getElementById('webcam1');
var webcamElement2 = document.getElementById('webcam2');
webcamElement1.style.visibility = 'hidden';
webcamElement2.style.visibility = 'hidden';
const crashHandlerCanvas = document.getElementById('webGL_crashHandler');
var webcam;     //create a webcam tf container for the tensorflow functions
let startMain;  //contains the main starting function
//set to around 300 at first
const base_recognizer = speechCommands.create('BROWSER_FFT');
const ModWidth = parseInt(document.getElementsByTagName('VIDEO')[0].getAttribute('width'));
const ModHeight = parseInt(document.getElementsByTagName('VIDEO')[0].getAttribute('height'));
if (ModHeight > 250 || ModWidth > 280) {
    alert('The set resolution of the camera is too high!');
}
//in the latest versions, the audio classifier doesn't have an original number
var audio_classifier_busy = false;
var restoredAudioClassifier = false;           //if we have restored the classifier data
var displayedOnceAudioClassifier = false;      //have we displayed the warning
var song_classifier_id;                        //the initial number of examples_SONG_ID in the array
var restoredSongIdentification = false;        //if we have restored the classifier
var displayedOnceSongIdentification = false;   //if we have displayed the warning once
var songID_busy = false;                       //store if we have songID_busy anything from the backend core
var continueLearn_songID = false;              //store if we have started learning the song
//if we have received any request from the backend
var speakerID_busy = false;                    //if speaker ID is doing some work
var speaker_classifier_index;                  //the initial number of examples_SPEAKER_ID in the array
var restoredSpeakerIdentification = false;
var displayedOnceSpeakerIdentification = false;
const classNum = 10;                           //no of examples to be trained for the visual classifier for a particular label
//if we have to modify the current visual recognition model
let visual_recognition_busy = false;
let displayedOnceVisualRecognition = false;    //if we have already displayed the warning once
var restoredFaceAPI = false;
//store if we have restored the face api model
var displayedOnceFaceAPI = false;
//if we have to modify the face api model
let modified_face_api = false;
//each data of the face API result looks like this
const oneFaceAPIData = {
    "name": null,
    "gender": null,
    "age": null,
    "neutral": null,
    "sadness": null,
    "joy": null,
    "fear": null,
    "disgust": null,
    "anger": null,
    "surprise": null
}
const minMicVol_AUDIO_CLASSIFIER = 10;         //min level
//setting up the looping delays
const maxDelayTime_AUDIO_CLASSIFIER = 4000;    //initially set to 4000, cannot exceed this time
const initialisationDelayTF = 500;
const faceapiIterationDelay = 3000;            //general iteration delay 
const faceapiLimitDelay = 4500;                //cannot exceed this time
const visualRecognitionIterationDelay = 1000;
const objectDetectionIterationDelay = 1000;
const poseDetectionIterationDelay = 1000;
const textToxicityDelay = 500;
const outputCommDelay = 500;
const communicationDelay = 500;            //the intervals after which the tf communication requests will be sent to the backend
//load all the models here
var verbalClassifier = knnClassifier.create();
var poseClassifier = knnClassifier.create();
var verbalClIndex = 0; //set to undefined
var poseClIndex = 0;   //set to undefined,
var visual_CLindex = 0;//set to undefined
var visual_classifier = knnClassifier.create();
var visual_net;
var coco;               //object detection model
var toxicDetector;      //text toxicity model
var pose_net;
var faceMatcher;        //the face matcher for custom face classification
var audio_classifier;
var song_classifier;
var speaker_classifier;

const jsonConf = {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
}

let useTinyModel = true;        //we are going to use the tiny model only
if (navigator.hardwareConcurrency > 4)
    console.info("The system is capable of running in advanced mode. However basic mode is defaulted.")

let ifConnectedTF = window.navigator.onLine;
let webcam2NotLoaded = true;
//crash handlers
crashHandlerCanvas.addEventListener('webglcontextlost', async function (event) {
    event.preventDefault();
    await audio_classifier.stopListening();
    await song_classifier.stopListening();
    await speaker_classifier.stopListening();
    clearTimeout(startMain)
}, false)

crashHandlerCanvas.addEventListener('webglcontextrestored', function () {
    startMain = setTimeout(mainTF, 50)
}, false);

//image search for visual recognition
gapi.load("client");
var testImg = document.getElementById('initImage');
testImg.style.visibility = 'hidden';    //we don't want to show the initialisation image

//the initialisation of all the models
webcamElement2.addEventListener('play', () => webcam2NotLoaded = false)
if (ifConnectedTF == true) {
    startVideo();     //start the video from the webcam
    startFaceAPIVideo();
    async function customFaceCl() {         //allow this to run separately and not in series
        console.log('Frontend TF: Initialising the face api custom mode.');
        const labeledDescriptors = await loadLabeledImages();
        faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.6);
        restoredFaceAPI = true;
        console.log('Frontend TF: Custom Face classifier restored');
        return
    }
    async function visualLoad() {
        visual_net = await mobilenet.load();
        console.log('Frontend TF: Pretrained Visual Classifier loaded.')
        return
    };
    async function objectLoad() {
        coco = await cocoSsd.load();
        console.log('Frontend TF: Coco Ssd Object detection model loaded.')
        return
    };
    async function toxicityLoad() {
        toxicDetector = await toxicity.load();
        console.log('Frontend TF: Text toxicity model successfully loaded.');
        return
    };
    //no advanced mode in the current version of RIO
    async function poseDetectLoad() {
        pose_net = await posenet.load({
            architecture: 'MobileNetV1',
            outputStride: 16,//32
            inputResolution: { width: ModWidth, height: ModHeight },
            multiplier: 0.75
            //quantBytes: 2
        });
        console.log('Frontend TF: Pose detection classifier model loaded');
        return
    };
    async function audioMediaClLoad() {
        //creating the transfer recognizers
        await base_recognizer.ensureModelLoaded();
        audio_classifier = base_recognizer.createTransfer('audio_classification');
        console.log('Frontend TF: Audio classifier model loaded.');
        song_classifier = base_recognizer.createTransfer('song_identification');
        console.log('Frontend TF: Song idenitification model loaded.');
        speaker_classifier = base_recognizer.createTransfer('speaker_identification');
        console.log('Frontend TF: Speaker Identification model loaded.');
        return
    };
    Promise.all([
        console.log('Frontend TF: Loading robotic intuition operator tensorflow models.'), ,
        //actually these directory structures represents the inside the file server app not the computer
        faceapi.nets.ageGenderNet.loadFromUri('tensorflow/face_api/models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('tensorflow/face_api/models'),
        faceapi.nets.faceExpressionNet.loadFromUri('tensorflow/face_api/models'),
        faceapi.nets.ssdMobilenetv1.loadFromUri('tensorflow/face_api/models'),
        faceapi.nets.faceLandmark68TinyNet.loadFromUri('tensorflow/face_api/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('tensorflow/face_api/models'),
        faceapi.nets.tinyFaceDetector.loadFromUri('tensorflow/face_api/models'),
        faceapi.nets.tinyYolov2.loadFromUri('tensorflow/face_api/models'),
        toxicityLoad(),
        visualLoad(),
        objectLoad(),
        poseDetectLoad(),
        audioMediaClLoad(),
    ])
        .then(() => customFaceCl())
        .then(() => {
            loadClient();     //load the custom search engine client
            console.log('Frontend TF: Face API models loaded. \nAll models have been loaded successfully');
        })
        .then(initAndTest)
}
else console.error('Frontend TF: Internet connection not available. Please connect to the Internet and try again!!');

//wait for the webcam element to be loaded
async function initFrontend() {
    console.info('Frontend TF: Please wait while robotic intuition operator engine is being loaded..')
    //wait until both video recorder and frontend javascript are ready

    let initReqTF = true;
    while (initReqTF == true) {
        await fetch('http://127.0.0.1:5000/initialisation', {
            headers: jsonConf,
            body: JSON.stringify({ module: 'frontend' }),
            method: 'POST'
        })
            .then(res => res.json())
            .then(data => {
                if (data.start == true) {
                    //fade out the loader
                    $(".loader").fadeOut("slow");
                    console.info('Frontend TF: Robotic Intuition Operator engine loaded successfully. Press the button to start')
                    //time to start the main thing when the user has clicked on the start button
                    startBtn.style.visibility = 'visible';       //show the button now
                    initReqTF = false;
                }
            })
            .catch(err => console.error('Frontend TF: tensorflow frontend: cannot connect to the initialisation server', err))
        await sleep(initialisationDelayTF);
    }

    //click when we are ready
    startBtn.addEventListener('click', function () {
        startBtn.style.visibility = 'hidden';    //hide the button once again the page has been loaded
        fetch('http://127.0.0.1:5000/startTheBackend', {
            headers: jsonConf,
            body: JSON.stringify({ message: "startBackend" }),
            method: 'POST'
        })
            .then(res => res.json())
            .then(data => {
                if (data.start == true) {
                    //after we have been allowed to fire!
                    startFrontend = true;
                    startMain = setTimeout(mainTF, 20);
                    sendOutput();   //keep sending the output in intervals of time
                }
                else console.error("Backend has not been allowed to start. Frontend will also not start.");
            })
    })
}

async function startFaceAPIVideo() {
    console.log('Frontend TF: Loading webcam source 2 face api webcam.');
    navigator.getUserMedia(
        { video: {} },
        stream => webcamElement2.srcObject = stream,
        err => console.error('Frontend TF: Error in loading webcam 2: ', err)
    )
    console.log('Frontend TF: Successfully loaded webcam 2 (face api webcam)');
}

async function startVideo() {
    console.log('Frontend TF: Loading webcam source 1.')
    //webcam 1 is for tensorflow visual functions
    try {
        webcam = await tf.data.webcam(webcamElement1);
        console.log('Frontend TF: Successfully loaded webcam 1 (TFjs visual webcam)')
    }
    catch (err) {
        console.error("Frontend TF: Error in loading webcam 1 (TFjs visual webcam): ", err.message);
    }
}

var prevFaceAPIStartTime;
var prevFaceAPIEndTime;
var faceAPIStartTime;
var faceAPIEndTime;
let firstIterationFACE = true;

//this is the main function of the entire program and also the most heavily loaded one
function mainTF() {
    communication();            //communications will run even in the test mode
    //the audio media classification modules will be controlled by the incoming modules themselves
    if (restoredAudioClassifier == false && displayedOnceAudioClassifier == false) {
        console.warn('The audio classifier has started without being restored.');
        displayedOnceAudioClassifier = true;
    }
    if (restoredSongIdentification == false && displayedOnceSongIdentification == false) {
        console.warn('Song identification classifier has started without being restored.');
        displayedOnceSongIdentification = true;
    }
    if (restoredSpeakerIdentification == false && displayedOnceSpeakerIdentification == false) {
        console.warn("Speaker identification classifier has started without being restored.")
        displayedOnceSpeakerIdentification = true;
    }
    if (testMode == false) {
        textToxicity();
        visualRecognition_app();
        faceAPI();
        objectDetection_app();
        poseDetection_app();
        listenNow_audioClassifier(true);        //we start the first occurance of the audio classifier
        listenNow_songID(true);                 //also of the song classifier
        listenNow_SpeakerID(true);
    }
}

//text toxicity detection, the newer version only sends the detected labels
async function textToxicity() {
    while (true) {
        const minCharactersInSpeech = 4;
        if (final_transcript.length >= minCharactersInSpeech) {
            var predictions = await toxicDetector.classify(final_transcript);
            if (predictions !== undefined) {
                var toxicity_matched = new Array(0);
                predictions.forEach(el => {
                    if (el.results[0].match == true) {
                        toxicity_matched.push(el.label + '_');              //it's a system stimulus if it is matched
                    }
                });
                if (toxicity_matched.length !== 0) {
                    if (outputData.textToxicity.data !== null) {
                        setTimeout(function () {
                            outputData.textToxicity.data = toxicity_matched;
                        }, outputCommDelay)
                    }
                    else outputData.textToxicity.data = toxicity_matched;
                }
            }
        }
        await sleep(textToxicityDelay);
    }
}

var onceDoneTensorflow = false;    //if we have already initialied the tensorflow functions
async function initAndTest() {
    restoreModels();    //restore the models now itself
    console.log('Frontend TF: Initialising and testing all rio models. Please wait this may take a while.');
    //firstly wait for the face api webcam to be loaded fully
    while (webcam2NotLoaded == true) await sleep(500);  //waiting for modules to be loaded won't create much stress I think
    if (onceDoneTensorflow == false) {
        onceDoneTensorflow = true;
        const audioMediaParams = {
            includeSpectrogram: false,
            probabilityThreshold: 0.75,
            overlapFactor: 0.999,
            invokeCallbackOnNoiseAndUnknown: true
        };
        var audioClRes, songIDRes, speakerID;
        console.log('Frontend TF: Testing the audio media classification modules..')
        await audio_classifier.listen(res => {
            audioClRes = res;
        }, audioMediaParams);
        await song_classifier.listen(res => {
            songIDRes = res;
        }, audioMediaParams);
        await speaker_classifier.listen(res => {
            speakerID = res;
        }, audioMediaParams);
        while (audioClRes == undefined || songIDRes == undefined || speakerID == undefined) {
            //just wait until the audio media classifier modules have been initialised
            await sleep(500);
        }
        //in the latest version we are just waiting for the audio modules to be loaded
        console.log('Frontend TF: Audio cl init response: ', JSON.stringify(audioClRes));
        console.log('Frontend TF: Song ID init response: ', JSON.stringify(songIDRes));
        console.log('Frontend TF: Speaker ID init response: ', JSON.stringify(speakerID))
        //stop listening to the sounds
        await audio_classifier.stopListening();
        await song_classifier.stopListening();
        await speaker_classifier.stopListening();
        //wait for the face api webcam element to be fully loaded
        console.log('Frontend TF: Audio media classification modules have been successfully loaded.');
        console.log('Frontend TF: Done...Testing Visual classifier model..');
        const visualDetectRes = await visual_net.classify(testImg);
        console.log('Frontend TF: Visual Recognition init response: ', JSON.stringify(visualDetectRes));
        console.log('Frontend TF: Done...Testing object detection model..');
        const objectDetectRes = await coco.detect(testImg);
        console.log('Frontend TF: Object Detection init response: ', JSON.stringify(objectDetectRes))
        console.log('Frontend TF: Done...Testing pose detection model..');
        const poseDetectRes = await pose_net.estimateSinglePose(testImg, {
            flipHorizontal: false
        });
        console.log('Frontend TF: Pose classifier init response: ', JSON.stringify(poseDetectRes));
        console.log('Frontend TF: Done...Testing face api model..')
        var faceDetectRes = undefined;
        //SsdMobilenetv1Options()
        //TinyFaceDetectorOptions()
        console.log('Frontend TF: Initialising Face API..');
        faceDetectRes = await faceapi.detectSingleFace(testImg, new faceapi.SsdMobilenetv1Options())
        //.withFaceLandmarks(useTinyModel)
        //.withFaceExpressions()
        //.withAgeAndGender()
        //.withFaceDescriptor()

        console.log('Frontend TF: Face API init response: ', JSON.stringify(faceDetectRes));
        function stringArrToNumeric(array) {
            //convert an array strings into an array of numbers, needed for custom classifier
            let numArray = new Array(0);
            array.forEach(el => {
                var numForm = 0;
                for (var i = 0; i < el.length; i++) {
                    numForm = numForm * 10 + el.charCodeAt(i);
                }
                numArray.push(numForm);
            });
            return numArray;
        }
        let verbalInitRawData = [
            stringArrToNumeric(["sad", "behaviour", "bad", "unexpected", "hurt", "rude", "sorry", "depressed"]),
            stringArrToNumeric(["I", "am", "very", "sad", "for", "what", "happened", "sorry", "really", "you"]),
            [0.75, 0.3, 0.4, 0.4, 0.3, 0.4],
            [0.75, 0.3, 0.3, 0.3, 0.3, 0.3]
        ];
        const verbalInitData = tf.tensor1d(verbalInitRawData.flat());
        let verbalInitOutput = verbalClassifier.predictClass(verbalInitData);
        let poseInitRawData = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        const poseInitData = tf.tensor1d(poseInitRawData.flat());
        let poseInitOutput = poseClassifier.predictClass(poseInitData);
        //the model load checker
        if (Object.keys(audioClRes).length !== 0 || audioClRes.constructor !== Object) {
            if (Object.keys(songIDRes).length !== 0 || songIDRes.constructor !== Object) {
                if (Object.keys(speakerID).length !== 0 || speakerID.constructor !== Object) {
                    if (visualDetectRes != undefined && objectDetectRes != undefined && poseDetectRes != undefined && faceDetectRes != undefined) {
                        if (Object.keys(visualDetectRes).length !== 0 || visualDetectRes.constructor !== Object) {
                            if (Object.keys(objectDetectRes).length !== 0 ||
                                objectDetectRes.constructor !== Object) {
                                if (Object.keys(poseDetectRes).length !== 0 || poseDetectRes.constructor !== Object) {
                                    if (Object.keys(faceDetectRes).length !== 0 || faceDetectRes.constructor !== Object) {
                                        if (Object.keys(verbalInitOutput).length !== 0 || verbalInitOutput.constructor !== Object) {
                                            if (Object.keys(poseInitOutput).length !== 0 || poseInitOutput.constructor !== Object) {
                                                console.log('Frontend TF: All RIO models have been successfully initialised.');
                                            }
                                            else console.error("Frontend TF: Error in initialising the pose classifier.")
                                        }
                                        else console.error("Frontend TF: Error in initialising verbal classifier")
                                    }
                                    else console.error('Frontend TF: Error in initialiding the face api function.');
                                }
                                else console.error('Frontend TF: Error in initialising pose detection function.');
                            }
                            else console.error('Frontend TF: Error in initialsing the object detection function.');
                        }
                        else console.error('Frontend TF: Error in initialising the visual recognition function.');
                    }
                    else console.error('Frontend TF: Error in initialising graphical functions.')
                }
                else console.error('Frontend TF: Error in initialising the speaker identification module.');
            }
            else console.error('Frontend TF: Error in initialising the song identification module.');
        }
        else console.error('Frontend TF: Error in initialising the audio classifier module.');
        //we start asking for init confirmation after the initialisation of the rio models
        console.log('Frontend TF: Initialisation and testing of models has been completed.');
        initFrontend();
        return;
    }
}

function restoreModels() {
    //one restore function for all the models
    $.post('http://127.0.0.1:5000/restore', async function (data, status) {
        if (status == 'success' && data !== undefined) {
            //verbal restore
            if (data.verbal_restore.dataset !== null) {
                restoreVerbalClassifier(data.verbal_restore.dataset);
                console.log('Frontend TF: Verbal classifier restored successfully.');
            }
            else console.warn('Frontend TF: Verbal restoration dataset is null.')

            //pose restore
            if (data.pose_restore.dataset !== null) {
                restorePoseClassifier(data.pose_restore.dataset);
                console.log('Frontend TF: Pose classifier restored successfully.');
            }
            else console.warn('Frontend TF: Pose restoration dataset is null.');

            //alternative pose restoration
            if (data.altPoseRestore.dataset !== null) {
                poseResponseAltSaved = data.altPoseRestore.dataset;
            }
            else console.warn("Alternative pose response dataset is null.");

            //audio restore
            if (data.audio_restore.dataset !== null) {
                restoration_audioClassifier(data.audio_restore.dataset);
            }
            else console.warn('Frontend TF: Audio restoration dataset is null.');

            //song restore
            if (data.song_restore.dataset !== null) {
                restoration_songID(data.song_restore.dataset);
            }
            else console.warn('Frontend TF: Song restoration dataset is null.');

            //speaker restore
            if (data.speaker_restore.dataset !== null) {
                restoration_speakerID(data.speaker_restore.dataset);
            }
            else console.warn('Frontend TF: Speaker restoration dataset is null.');

            //visual restore
            if (data.visual_restore.dataset !== null) {
                restoreDatasetVisualRec(data.visual_restore.dataset);
            }
            else console.warn('Frontend TF: Custom visual dataset is null');
        }
    })
}

let displayedOncePoseClassifier = false;
//all communication features have been incorporated into this function
async function communication() {
    $(async function () {
        //this is the continuous communication function
        while (true) {
            await sleep(communicationDelay);   //give some breathing room by waiting for a sufficient time before communication requests
            //in the current version we don't have face api modification
            await $.post('http://127.0.0.1:5000/input', async function (data, status) {
                if (status == 'success' && data != undefined) {
                    //verbal incoming
                    switch (data.verbal_incoming.text) {
                        case "learn":
                            if (data.verbal_incoming.json.length == 0)
                                console.error("Verbal Learn: input data has 0 length.");
                            VerbalLearn(data.verbal_incoming.json)
                            break;
                        case "predict":
                            predictVerbalAnswers(data.verbal_incoming.json)
                            break;
                        case "saveFinal": saveVerbalAnswers();
                            break;
                        case "savePeriodic": saveVerbalAnswers('savePeriodic');
                            break;
                    }

                    //pose incoming
                    console.log("Pose incoming text: ", data.pose_incoming.text)
                    switch (data.pose_incoming.text) {
                        case "learn":
                            if (data.pose_incoming.json.length == 0)
                                console.error("Pose Learn: input data has 0 length");
                            poseLearn(data.pose_incoming.json);
                            break;
                        case "predict":
                            if (poseClIndex !== undefined && poseClIndex !== 0) //if we don't have an empty dataset
                                predictPose(data.pose_incoming.json);
                            //warn once to the user if we do
                            else if (displayedOncePoseClassifier == false) {
                                console.warn('Pose classification cannot be done because the dataset is empty.');
                                displayedOncePoseClassifier = true;
                            }
                            break;
                        case "saveFinal":
                            savePose('saveFinal');
                            break;
                        case "savePeriodic":
                            savePose('savePeriodic');
                            break;
                    }

                    //in the test mode the stimulus detection functions will be halted
                    if (testMode == false) {
                        //visual incoming
                        switch (data.visual_incoming.text) {
                            case 'deleteSingle':
                                visual_recognition_busy = true;
                                visual_classifier.clearClass(data.visual_incoming.val);//delete the single class provided
                                visual_recognition_busy = false;
                                break;
                            //delete all the classes learnt till now
                            //Warning!! This is a very dangerous command. Handle with care
                            case 'deleteAll':
                                visual_recognition_busy = true;
                                visual_classifier.clearAllClasess();
                                visual_recognition_busy = false;
                                break;
                            //special case scenario when the backend tells us to learn images for the detected stimulus
                            case 'learnStimulus':
                                visual_recognition_busy = true;
                                if (visual_CLindex == undefined) {
                                    console.warn("visual_CLindex is undefined in learnStimulus section of communcication(). Resetting to 0");
                                    visual_CLindex = 0;
                                }
                                visualLearnStimulus(data.visual_incoming.val, visual_CLindex);
                                visual_CLindex++;
                                console.log("Learnt visual image with index ", visual_CLindex);
                                break;
                            case 'learn':
                                visual_recognition_busy = true;
                                //learn when the backend tells us                            
                                async function addExample(classId) {
                                    //get 10 images for the current learning index
                                    for (var m = 0; m < classNum; m++) {
                                        const img = await webcam.capture();
                                        const activation = visual_net.infer(img, 'conv_preds');
                                        visual_classifier.addExample(activation, classId);
                                        console.log("Learnt general visual image with index: ", classId)
                                        img.dispose();
                                    }
                                    visual_recognition_busy = false;
                                }
                                if (visual_CLindex == undefined) {
                                    console.error("visual_CLindex is undefined in learn section of communication()");
                                    visual_CLindex = 0;
                                }
                                addExample(visual_CLindex);
                                visual_CLindex++;
                                break;
                            case 'saveFinal':                           //saving would be done at the start once
                                saveVisualDataset('saveFinal');         //synchronous save
                                break;
                            case 'savePeriodic':
                                saveVisualDataset('savePeriodic')       //asynchronous save
                                break;
                        }

                        //audio incoming
                        switch (data.audio_incoming.text) {
                            case 'learn':
                                audio_classifier_busy = true;
                                //make sure we are not listening while learning
                                if (audio_classifier.isListening()) await audio_classifier.stopListening()
                                learnWithLoudness_audioClassifier(data.audio_incoming.val);
                                break;
                            case 'saveFinal':
                                //send the dataset to the backend for saving
                                saving_audioClassifier('saveFinal');
                                break;
                            case 'savePeriodic':
                                saving_audioClassifier('savePeriodic');
                                break;
                            default:
                                //then we just do on listening to the sounds
                                if (audio_classifier_busy == false)
                                    if (audio_classifier.isListening() == false)
                                        listenNow_audioClassifier(true);
                        }

                        //song incoming
                        switch (data.song_incoming.text) {
                            case 'learnStart':
                                songID_busy = true;
                                if (song_classifier.isListening())
                                    await song_classifier.stopListening()
                                learn_songID();
                                break;
                            case "learnStop":
                                //learning stops with this variable
                                continueLearn_songID = false;
                                break;
                            case 'deleteClass':
                                //we now delete the last and recent example
                                try {
                                    if (song_classifier.isListening())
                                        await song_classifier.stopListening()
                                    //classifier cannot have undefined index when deleting the class
                                    if (song_classifier_id != undefined) {
                                        song_classifier.clearClass(song_classifier_id);
                                        song_classifier_id--;
                                        console.log('Frontend TF: Successfully deleted the last song classifier example.');
                                    }
                                    else console.error("Frontend TF: cannot delete song classifier example when song index is undefined.");
                                    songID_busy = false;
                                }
                                catch (error) {
                                    console.error("Error in deleting song classifier class: ", error.message)
                                }
                                break;
                            case 'saveClass':
                                //when told to save the song learnt
                                try {
                                    await song_classifier.train({
                                        epochs: 25,
                                        callback: {
                                            onEpochEnd: async (epoch, logs) => {
                                                console.log(`Song ID Epoch ${epoch}: loss=${logs.loss}, accuracy=${logs.acc}`);
                                            }
                                        }
                                    });
                                }
                                catch (error) {
                                    console.error("Frontend TF: Speaker classifier saveClass training error: ", error.message)
                                }
                                console.log("Frontend TF: Song classifier learnt new song successfully");
                                songID_busy = false;
                                break;
                            case 'saveFinal':
                                saving_songID('saveFinal');
                                break;
                            case 'savePeriodic':
                                saving_songID('savePeriodic');
                                break;
                            default:
                                if (songID_busy == false)
                                    if (song_classifier.isListening() == false)
                                        listenNow_songID(true);
                        }

                        //speaker incoming
                        switch (data.speaker_incoming.text) {
                            case 'learn':
                                speakerID_busy = true;
                                if (speaker_classifier.isListening())
                                    await speaker_classifier.stopListening()
                                learn_speakerID();
                                break;
                            case 'saveFinal':
                                saving_speakerID('saveFinal');
                                break;
                            case 'savePeriodic':
                                saving_speakerID('savePeriodic');
                            default:
                                if (speakerID_busy == false)
                                    if (speaker_classifier.isListening() == false)
                                        listenNow_SpeakerID(true);
                        }
                    }

                    //frontend general communications
                    if (data.frontend_data.speech !== null) {
                        responseText = data.frontend_data.speech;
                        if (responseText.length > 0)
                            console.log("Frontend TF: response from backend: ", responseText)
                    }
                    if (data.frontend_data.emotion !== null) {
                        current_emotion = data.frontend_data.emotion;
                        console.log("Frontend TF: emotion from backend: ", current_emotion);
                    }
                    if (data.frontend_data.rate !== null) {
                        rate = String(data.frontend_data.rate);
                        console.log("Frontend TF: rate from backend: ", rate);
                    }
                    else rate = defaultSpeed;
                    if (data.frontend_data.pitch !== null)
                        pitch = String(data.frontend_data.pitch);
                    else pitch = defaultPitch;
                    if (data.frontend_data.singingState !== null) {
                        singingState = data.frontend_data.singingState;
                        console.log("Frontend TF: singingState from backend: ", singingState);
                    }
                    //frontend javascript song incoming communications
                    if (data.song_incomingJS.text !== null) {
                        songIncomingMessage = data.song_incomingJS.text;
                        console.log("Frontend TF: songIncomingMessage from backend: ", songIncomingMessage);
                    }
                }
            });
        }
    })
}
//----------------------------Object recognition model-----------------------------------
async function objectDetection_app() {
    while (true) {
        const img = await webcam.capture();
        const predictions = await coco.detect(img);
        //extract the objects and score from the result
        var detectedObjects = new Array(0);
        var objectScores = new Array(0);
        var x;
        //need to modify this part after checking
        for (x in predictions) {
            detectedObjects.push(predictions[x].class);
            objectScores.push(predictions[x].score);
        }
        //remove the unsure elements from the array, we aren't including this in the current version
        function getConfident(objects, score) {
            let conArray = new Array(0);
            objects.forEach((el, i) => {
                if (score[i] >= 4)
                    conArray.push(el);
            })
            return conArray;
        }
        if (outputData.objectDetect.data !== null) {
            setTimeout(function () {
                outputData.objectDetect.data = detectedObjects;
            }, outputCommDelay)
        }
        else outputData.objectDetect.data = detectedObjects;
        var outputString = '';
        var outputScore = '';
        outputString = detectedObjects.join(" ");
        outputScore = objectScores.join(" ");
        if (predictions.length != undefined)
            document.getElementById('object_detection').innerText = `
                        detected objects: ${outputString}\n
                        object scores:${outputScore}`;
        img.dispose();
        await tf.nextFrame();
        //give some breathing room for waiting for the next frame
        await sleep(objectDetectionIterationDelay);
    }
}

//------------------------------Pose detection model-------------------------------------
async function poseDetection_app() {
    //order of the parts in the array------
    //nose                  //leftHip
    //leftEye               //rightHip
    //rightEye              //leftKnee
    //leftEar               //rightKnee
    //rightEar              //leftAnkle
    //leftShoulder          //rightAnkle
    //rightShoulder         
    //leftElbow
    //rightElbow
    //leftWrist
    //rightWrist
    while (true) {
        const img = await webcam.capture();
        const pose = await pose_net.estimateSinglePose(img, {
            flipHorizontal: false
        });
        var nosePositionX = pose.keypoints[0].position.x;
        var nosePositionY = pose.keypoints[0].position.y;
        //convert response into JSON and and send it to the backend
        if (pose !== undefined) {
            let servoDegrees = analyze_poseDetection(pose)
            document.getElementById('pose_detection').innerHTML = `
            head horizontal servo: ${servoDegrees[0]} <br>
            right shoulder vertical servo: ${servoDegrees[2]} <br>
            left shoulder vertical servo: ${servoDegrees[3]} <br>
            right shoulder horizontal servo: ${servoDegrees[4]} <br>
            left shoulder horizontal servo: ${servoDegrees[5]} <br>
            right elbow servo: ${servoDegrees[6]} <br>
            left elbow servo: ${servoDegrees[7]} <br>
            base servo: ${servoDegrees[8]} <br>
            `;
            if (outputData.poseDetect.data !== null) {
                setTimeout(function () {
                    outputData.poseDetect.data = servoDegrees;
                    outputData.poseDetect.faceTracker.data = {
                        x: nosePositionX,
                        y: nosePositionY
                    }
                }, outputCommDelay)
            }
            else {
                outputData.poseDetect.data = servoDegrees;
                outputData.poseDetect.faceTracker.data = {
                    x: nosePositionX,
                    y: nosePositionY
                }
            }
        }

        img.dispose();
        await tf.nextFrame();
        await sleep(poseDetectionIterationDelay)
    }
}
//----------------------------The audio classifier module--------------------------------------
async function learnWithLoudness_audioClassifier(givenEmotion) {
    var lastHeardTime = new Date().getTime();
    var currentTime = lastHeardTime;
    var continueLearning = true;
    //continue learning until we keep hearing the sound
    console.log("Learning snippets for audio classifier.");
    try {
        while (continueLearning == true) {
            currentTime = new Date().getTime();
            //add the sound if it is greater than the minimum limit
            if (micVol >= minMicVol_AUDIO_CLASSIFIER) {
                await audio_classifier.collectExample(givenEmotion);
                lastHeardTime = new Date().getTime();
            }
            //the loop exit control mechanism
            else if (currentTime - lastHeardTime >= maxDelayTime_AUDIO_CLASSIFIER) {
                continueLearning = false;
            }
        }
        console.log("Sounds collected. Training the audio classifier.")
        //actually train the classifier now
        await audio_classifier.train({
            epochs: 25,
            callback: {
                onEpochEnd: async (epoch, logs) => {
                    console.log(`Audio Classifier Epoch ${epoch}: loss=${logs.loss}, accuracy=${logs.acc}`);
                }
            }
        });
        console.log("Frontend TF: Audio classifier sound learnt successfully");
    }
    catch (err) {
        console.error("Frontend TF: Audio classifier learn error: ", err.message)
    }
    audio_classifier_busy = false;
}

//bubble sort the given array
function bubble_sort(arr, score) {
    var length = arr.length, stop;
    for (var i = 0; i < length; i++) {
        for (var j = 0, stop = length - i; j < stop; j++) {
            if (score[j] > score[j + 1])
                swap(arr, j, j + 1);
        }
    }
}

//swapping of the two data
function swap(arr, firstIndex, secondIndex) {
    var temp = arr[firstIndex];
    arr[secondIndex] = arr[firstIndex];
    arr[firstIndex] = temp;
}

//listen to the sounds now
function listenNow_audioClassifier(sendToBackend, displayResults) {
    //true if we want to send data to backend
    audio_classifier.listen(result => {
        var detected_labels = audio_classifier.wordLabels();
        bubble_sort(detected_labels, result.scores);
        const label = detected_labels[0];               //get the sound with the highest confidence
        if (displayResults == undefined || displayResults == true)
            document.getElementById('audio_classifier').innerText = label;
        if (sendToBackend == true) {
            if (outputData.audioObjects.data !== null) {
                setTimeout(function () {
                    outputData.audioObjects.data = label
                }, outputCommDelay)
            }
            else {
                outputData.audioObjects.data = label;       //include the output sound in the output object
            }
        }
    }, {
        includeSpectrogram: false,
        probabilityThreshold: 0.75,
        overlapFactor: 0.999,
        invokeCallbackOnNoiseAndUnknown: true
    });
}

//restoration of the audio classifier
async function restoration_audioClassifier(dataset) {
    const clearExisting = false;
    const serializedDataset = codec.decode(dataset);
    console.log("Audio classifier dataset: ", serializedDataset)
    await audio_classifier.loadExamples(serializedDataset, clearExisting);
    await audio_classifier.train({
        epochs: 25,
        callback: {
            onEpochEnd: async (epoch, logs) => {
                console.log(`Audio classifier Epoch ${epoch}: loss=${logs.loss}, accuracy=${logs.acc}`);
            }
        }
    })
        .catch(err => console.error("Error in restoring the audio classifier: ", err));
    restoredAudioClassifier = true;
    console.log('Frontend TF: Custom audio classifier restored successfully with following data.', audio_classifier.countExamples());
}

function saving_audioClassifier(type) {
    if (type == undefined)
        type = 'saveFinal';
    //say the final good bye for now
    if (type == 'saveFinal') {
        responseText = "See you later.";
    }
    let base64Dataset;
    try {
        const serializedDataset = audio_classifier.serializeExamples();
        base64Dataset = codec.encode(serializedDataset);
    }
    catch (err) {
        console.warn("Audio classifier dataset is empty and thus cannot be saved.");
        base64Dataset = null;
    }
    let dataToSave = {
        audioSave: {
            data: base64Dataset,
            type: type
        }
    }
    fetch('http://127.0.0.1:5000/save', {
        headers: jsonConf,
        method: "POST",
        body: JSON.stringify(dataToSave)
    })
        .then(() => {
            console.log("Audio classifier save dataset sent to the backend.")
        })
        .catch(err => {
            console.log("Frontend TF: Error sending audio classifier save data: ", err);
        });
}
//----------------------------Song identification Module-------------------------------------
//the function for actually listening and recognising the sounds
function listenNow_songID(sendToBackend) {
    //true if we want to send data to backend
    //false if er don't want it
    song_classifier.listen(result => {
        //get all the labels that have been recognised
        var detected_labels = song_classifier.wordLabels();
        //we might not require this function to be here
        bubble_sort(detected_labels, result.scores);
        //get the sound with the highest confidence
        const label = detected_labels[0];
        document.getElementById('song_identification').innerText = label;
        if (sendToBackend == true) {
            if (outputData.songResponse.data !== null) {
                setTimeout(function () {
                    outputData.songResponse.data = label;
                }, outputCommDelay)
            }
            else
                outputData.songResponse.data = label;

        }
    }, {
        includeSpectrogram: false,
        probabilityThreshold: 0.75,
        overlapFactor: 0.999,
        invokeCallbackOnNoiseAndUnknown: true
    });
}

//restoration of the song identification module
async function restoration_songID(dataset) {
    const clearExisting = false;
    const serializedDataset = codec.decode(dataset)
    await song_classifier.loadExamples(serializedDataset, clearExisting);
    await song_classifier.train({
        epochs: 25,
        callback: {
            onEpochEnd: async (epoch, logs) => {
                console.log(`Song ID Epoch ${epoch}: loss=${logs.loss}, accuracy=${logs.acc}`);
            }
        }
    });
    let examplesCount = song_classifier.countExamples();
    song_classifier_id = Object.keys(examplesCount).length - 1;
    //get the last index, we don't necessarily depend upon the order of the numbers here
    console.log("Frontend TF: Song classifier restored successfully with following data: ", examplesCount);
    restoredSongIdentification = true;
}

//send the save request to the backend
function saving_songID(type) {
    if (type == undefined)
        type = 'saveFinal';
    let base64Serialized;
    try {
        const serialized = song_classifier.serializeExamples();
        base64Serialized = codec.encode(serialized);
    }
    catch (err) {
        console.warn("Song classifier dataset is empty and cannot be restored.");
        base64Serialized = null;
    }
    let dataToSave = {
        songSave: {
            data: base64Serialized,
            type: type
        }
    }
    //send the save request to the backend
    fetch('http://127.0.0.1:5000/save', {
        headers: jsonConf,
        method: "POST",
        body: JSON.stringify(dataToSave)
    })
        .then(() => {
            console.log('Sent song classifier save dataset to the backend.');
        })
        .catch(err => console.log("Frontend TF: Error sending save data: " + err));
}

async function learn_songID() {
    if (song_classifier_id == undefined) {
        console.warn("Song classifier index is undefined for the current song. Setting to 0")
        song_classifier_id = 0;
    }
    else {
        song_classifier_id++;
    }
    continueLearn_songID = true
    try {
        //continue learning until continueLearn_songID has been set to false in the communication module
        while (continueLearn_songID == true) await song_classifier.collectExample(String(song_classifier_id));
        //training of the song classifier will be done only when the user has confirmed the singing
    }
    catch (error) {
        console.error("Frontend TF: Error in collecting song learning examples: ", error.message)
    }
}

//the function for actually listening and recognising the sounds
function listenNow_SpeakerID(sendToBackend) {
    speaker_classifier.listen(result => {
        //the callback here is a loop
        //get all the labels that have been recognised
        var detected_labels = speaker_classifier.wordLabels();
        //we might not require this function to be here
        bubble_sort(detected_labels, result.scores);
        //get the sound with the highest confidence
        const label = detected_labels[0];
        document.getElementById('speaker_identification').innerText = label;
        if (sendToBackend == true) {
            if (outputData.speakerResponse.data !== null) {
                setTimeout(function () {
                    outputData.speakerResponse.data = label;
                }, outputCommDelay)
            }
            else outputData.speakerResponse.data = label;

        }

    }, {
        includeSpectrogram: false,
        probabilityThreshold: 0.75,
        overlapFactor: 0.999,
        invokeCallbackOnNoiseAndUnknown: true
    });
}

async function learn_speakerID() {
    if (speaker_classifier_index == undefined) {
        console.warn("Frontend TF: speaker classfier index in learn_speakerID() is undefined. Resetting to 0.");
        speaker_classifier_index = 0;
    }
    else {
        speaker_classifier_index++;
    }
    var lastHeardTime = new Date().getTime();
    var currentTime = lastHeardTime;
    var continueLearning = true;
    //continue learning until we keep hearing the sound
    console.log("Learning snippets for speaker classifier.");
    try {
        while (continueLearning == true) {
            currentTime = new Date().getTime();
            //add the sound if it is greater than the minimum limit
            if (micVol >= minMicVol_AUDIO_CLASSIFIER) {
                await speaker_classifier.collectExample(String(speaker_classifier_index));
                lastHeardTime = new Date().getTime();
            }
            //the loop exit control mechanism
            else if (currentTime - lastHeardTime >= maxDelayTime_AUDIO_CLASSIFIER) {
                continueLearning = false;
            }
        }
        console.log("Sounds collected. Training the speaker classifier.")
        //actually train the classifier now
        await speaker_classifier.train({
            epochs: 25,
            callback: {
                onEpochEnd: async (epoch, logs) => {
                    console.log(`Speaker ID Epoch ${epoch}: loss=${logs.loss}, accuracy=${logs.acc}`);
                }
            }
        });
        console.log("Frontend TF: Speaker classifier sound learnt successfully");
    }
    catch (err) {
        console.error("Frontend TF: Speaker classifier sound learn error: ", err.message);
    }
    speakerID_busy = false;
}

//restoration of the song identification module
async function restoration_speakerID(dataset) {
    const clearExisting = false;
    const serializedDataset = codec.decode(dataset);
    await speaker_classifier.loadExamples(serializedDataset, clearExisting);
    await speaker_classifier.train({
        epochs: 25,
        callback: {
            onEpochEnd: async (epoch, logs) => {
                console.log(`Speaker ID Epoch ${epoch}: loss=${logs.loss}, accuracy=${logs.acc}`);
            }
        }
    });
    const examplesCount = speaker_classifier.countExamples();
    restoredSpeakerIdentification = true;
    console.log("Speaker classifier restored with the following data: ", examplesCount)
    speaker_classifier_index = Object.keys(examplesCount).length - 1;
}
//send the save request to the backend
function saving_speakerID(type) {
    if (type == undefined) type = 'saveFinal';
    let base64Dataset;
    try {
        const serialized = speaker_classifier.serializeExamples();
        base64Dataset = codec.encode(serialized);
    }
    catch (err) {
        console.warn("Speaker classifier dataset is empty and cannot be restored.");
        base64Dataset = null;
    }
    let dataToSave = {
        speakerSave: {
            data: base64Dataset,
            type: type
        }
    }
    fetch('http://127.0.0.1:5000/save', {
        headers: jsonConf,
        method: "POST",
        body: JSON.stringify(dataToSave)
    })
        .then(() => {
            console.log("Frontend TF: Sent speaker classifier dataset to the backend.");
        })
        .catch(err => {
            console.log("Frontend TF: Error sending save data: " + err);
        });
}
//----------------------------Visual Recognition Module------------------------------------
async function visualRecognition_app() {
    while (true) {
        //custom visual objects, pretrained one has been removed
        if (visual_recognition_busy == false) {
            if (visual_CLindex != undefined && visual_CLindex !== 0) {
                const img = await webcam.capture();
                // Get the activation from mobilenet from the webcam.
                const activation = visual_net.infer(img, 'conv_preds');
                // Get the most likely class and confidence from the classifier module.
                const result = await visual_classifier.predictClass(activation);
                //we are just getting one output, we don't need to filter that anymore, and also we don't need the score                
                if (result != undefined) {
                    //send the detected visual object to the backend now
                    if (outputData.visualObjectsCustom.data !== null) {
                        setTimeout(function () {
                            outputData.visualObjectsCustom.data = result.label;
                        }, outputCommDelay)
                    }
                    else outputData.visualObjectsCustom.data = result.label;
                    document.getElementById('visual_recognition_custom').innerText = `Objects: ${result.label}`;
                }
                img.dispose();
                await tf.nextFrame();
            }
            else {
                //if we have not restored the data and have not displayed the warning
                if (displayedOnceVisualRecognition == false) {
                    console.warn('Custom visual classifier has not been restored.');
                    document.getElementById('visual_recognition_custom').innerText = `Visual Custom Classifier has not been restored.`
                    displayedOnceVisualRecognition = true;
                }
            }
        }
        //wait before moving to the next iteration
        await sleep(visualRecognitionIterationDelay);
    }
}

//visual classifier dataset save
function saveVisualDataset(type) {
    if (type == undefined) type = 'saveFinal'
    //when signalled get the full dataset for saving state
    let datasetObj = {};
    if (visual_CLindex !== undefined && visual_CLindex != 0) {
        var dataset = visual_classifier.getClassifierDataset();
        Object.keys(dataset).forEach((key) => {
            let data = dataset[key].dataSync();
            datasetObj[key] = Array.from(data);
        });
    }
    else {
        datasetObj = null;
    }
    let dataToSave = {
        visualSave: {
            data: datasetObj,
            type: type
        }
    }
    //send the save request to the backend
    fetch('http://127.0.0.1:5000/save', {
        headers: jsonConf,
        method: "POST",
        body: JSON.stringify(dataToSave)
    })
        .then(() => {
            console.log("Sent visual save dataset to the backend");
        })
        .catch(err => console.log("Frontend TF: Error sending visual save data: " + err));
}

function restoreDatasetVisualRec(data) {
    //when signalled restore a model
    var tensorObj = data;
    Object.keys(tensorObj).forEach((key) => {
        tensorObj[key] = tf.tensor(tensorObj[key], [tensorObj[key].length / 1024, 1024]);
    })
    visual_classifier.setClassifierDataset(tensorObj);
    console.log('Frontend TF: Visual dataset successfully restored. Custom visual recognition will now function efficiently.');
    visual_CLindex = visual_classifier.getNumClasses();           //count the number of total classes
    if (visual_CLindex == undefined) {
        console.error("visual_CLindex is undefined in restoreDatasetVisualRec().");
        visual_CLindex = 0;
    }
}

//------------------------------Face API module--------------------------------------------
async function faceAPI() {
    var faceAPP = setTimeout(faceAPI_app, 50);
    while (faceAPIStartTime == undefined || faceAPIEndTime == undefined) {
        //we just wait for it to be initialised
        await sleep(50);
    }

    while (true) {
        if (firstIterationFACE == false) {
            //we keep checking if face api is taking longer than usual
            if (faceAPIEndTime - faceAPIStartTime >= faceapiLimitDelay) {
                clearTimeout(faceAPP);
                console.log('Frontend TF: Face API has been reset for taking too long.')
                await sleep(3000);      //wait 3 seconds before starting again
                //we stopped the function and the corresponding loop
                break;
            }
            else {
                prevFaceAPIStartTime = faceAPIStartTime;
                prevFaceAPIEndTime = faceAPIEndTime;
            }
        }
        await sleep(1000);
    }
    faceAPI();
    return;
}

async function faceAPI_app() {
    while (true) {
        if (modified_face_api == false) {
            faceAPIStartTime = new Date().getTime();   //get starting time for this iteration
            var result = undefined;
            result = await faceapi.detectSingleFace(webcamElement2, new faceapi.SsdMobilenetv1Options())
                .withFaceLandmarks(useTinyModel)
                .withFaceExpressions()
                .withAgeAndGender()
                .withFaceDescriptor()
            var pretrained_display = "";
            var custom_display = "";
            var detected = new Array(0);
            var sendData = new Array(0);
            if (result != undefined) {
                sendData[0] = oneFaceAPIData;
                if (restoredFaceAPI == true) {
                    const bestMatch = faceMatcher.findBestMatch(result.descriptor)  //we find the best match
                    if (bestMatch != undefined) {
                        detected[0] = String(bestMatch);
                        custom_display += detected[0] + " ";
                        //replace with the name of the person
                        sendData[0].name = detected[0];
                    }
                }
                else {
                    if (displayedOnceFaceAPI == false) {
                        document.getElementById('face_custom').innerText = 'Custom Face Classifier not restored.'
                        console.warn('The custom face classifier has not been restored.')
                        displayedOnceFaceAPI = true;
                    }
                }
                sendData[0].neutral = result.expressions.neutral;
                sendData[0].anger = result.expressions.angry;
                sendData[0].sadness = result.expressions.sad;
                sendData[0].joy = result.expressions.happy;
                sendData[0].fear = result.expressions.fearful;
                sendData[0].disgust = result.expressions.disgusted;
                sendData[0].surprise = result.expressions.surprised;
                sendData[0].gender = result.gender;
                sendData[0].age = result.age;
                pretrained_display = `\nAge: ${sendData[0].age} \n
                                        Gender: ${sendData[0].gender} \n
                                        Neutral: ${sendData[0].neutral} \n
                                        Anger: ${sendData[0].anger} \n
                                        Sadness: ${sendData[0].sadness} \n
                                        Joy: ${sendData[0].joy} \n
                                        Fear: ${sendData[0].fear} \n
                                        Disgust: ${sendData[0].disgust} \n
                                        Surprise: ${sendData[0].surprise} \n\n`
                //program is working till here
                document.getElementById('face_pretrained').innerText = pretrained_display;
                if (restoredFaceAPI == true && detected[0] !== undefined) {
                    document.getElementById('face_custom').innerText = custom_display;
                }
                //send detections to backend
                if (sendData.length !== 0) {
                    if (outputData.faceAPI.data !== null) {
                        setTimeout(function () {
                            outputData.faceAPI.data = sendData;
                        }, outputCommDelay)
                    }
                    else outputData.faceAPI.data = sendData;
                }
                faceAPIEndTime = new Date().getTime();
            }
            //check if this is the first iteration
            if (firstIterationFACE == true)
                firstIterationFACE = false;
        }
        else modified_face_api = false;
        await sleep(faceapiIterationDelay);
    }
}

async function loadLabeledImages() {
    //the custom face classifier function
    return fetch('emotionalObjects/learnt_modules/labeledImages/names.json')
        .then(res => res.json())
        .then(jsonData => {
            let names = new Array(0);
            jsonData.names.forEach(el => {
                names.push(el);
            });
            return names;
        })
        .then(labels => {
            return Promise.all(
                labels.map(async label => {
                    let descriptions = []
                    for (let i = 1; i <= 2; i++) {  //maximum of 5 descriptors is allowed in the current version
                        //2 descriptors for two persons are used currently
                        const img = await faceapi.fetchImage(`emotionalObjects/learnt_modules/labeledImages/${label}/${i}.jpg`)
                            .catch(() => { return undefined })
                        if (img != undefined) {
                            const detections = await faceapi.detectSingleFace(img, new faceapi.SsdMobilenetv1Options())
                                .withFaceLandmarks(useTinyModel)
                                .withFaceDescriptor()
                            if (detections !== undefined)
                                descriptions.push(detections.descriptor)
                            else console.error('Frontend TF: Error in labelling faces for ', label);
                        }
                        else break;
                    }
                    return new faceapi.LabeledFaceDescriptors(label, descriptions)
                })
            )
        })
        .catch(err => console.error('Frontend TF: Error retrieving labeled images: ', err))
}

//pose classifier for the backend
function restorePoseClassifier(data) {
    var tensorObj = data;
    Object.keys(tensorObj).forEach((key) => {
        tensorObj[key] = tf.tensor(tensorObj[key], [1, tensorObj[key].length]);
    })
    console.log("Frontend TF: Restored pose classifier");
    poseClassifier.setClassifierDataset(tensorObj);
    poseClIndex = poseClassifier.getNumClasses();           //the index is always made less than 1
    if (poseClIndex == undefined) {
        console.error("poseClIndex is undefined in restorePoseClassifier().");
        poseClIndex = 0;
    }
}

//save the pose classifier
function savePose(type) {
    if (type == undefined) type = 'saveFinal';
    //otherwise savePeriodic is sent
    let datasetObj = {};
    if (poseClIndex !== undefined && poseClIndex !== 0) {
        var dataset = poseClassifier.getClassifierDataset();
        Object.keys(dataset).forEach((key) => {
            let data = dataset[key].dataSync();
            datasetObj[key] = Array.from(data);
        });
    }
    else {
        datasetObj = null;
    }
    let dataToSave = {
        poseSave: {
            data: datasetObj,
            type: type
        }
    }
    //time to save
    fetch('http://127.0.0.1:5000/save', {
        headers: jsonConf,
        method: 'POST',
        body: JSON.stringify(dataToSave)
    })
        .then(() => {
            console.log("Frontend TF: Sent pose classifier save dataset.");
        })
        .catch(err => console.error('Frontend TF: Error sending pose save data: ', err))
}

function predictPose(data) {
    //data here is the array of emotions of the current state
    const poseData = tf.tensor1d(data.flat());
    poseClassifier.predictClass(poseData)
        .then(classId => {
            let outputPoseResponse = classId.label;
            //we have an extra checking function using weighted average to find the nearest possible distance
            function summation(givenArray) {
                let sum = 0;
                for (var dh = 0; dh < givenArray.length; dh++) {
                    sum += givenArray[dh];
                }
                return sum;
            }
            let confidenceArray = [];           //array to store the confidences of the pose predicted
            const uncertainValue = 0.3333333333333333;
            let uncertainValues = 0;            //number of uncertain scores
            for (var x in classId.confidences) {
                confidenceArray.push(classId.confidences[x]);
            }
            for (var dh = 0; dh < confidenceArray.length; dh++) {
                if (confidenceArray[dh] == uncertainValue) uncertainValues++;
            }
            if (uncertainValues >= confidenceArray.length / 2) {
                //if we have a considerable number of uncertain values
                let distanceArrays = [];
                for (var db = 0; db < poseResponseAltSaved.length; db++) {
                    let basicEmDiff = [];       //absolute difference of all the basic emotions
                    for (var dh = 0; dh < poseResponseAltSaved[db].length; dh++) {
                        let absDiff = Math.abs(poseResponseAltSaved[db][dh] - data[dh]);
                        basicEmDiff.push(absDiff)
                    }
                    distanceArrays.push(summation(basicEmDiff));
                }
                let smallestIndex = 0;
                let smallestValue = distanceArrays[0];
                for (var kg = 1; kg < distanceArrays.length; kg++) {
                    if (distanceArrays[kg] < smallestValue) {
                        smallestValue = distanceArrays[kg];
                        smallestIndex = kg;
                    }
                }
                outputPoseResponse = String(smallestIndex);
            }
            if (outputData.poseResponse.type !== null) {
                setTimeout(function () {
                    outputData.poseResponse.type = "response";
                    outputData.poseResponse.data = outputPoseResponse;
                }, outputCommDelay)
            }
            else {
                outputData.poseResponse.type = "response";
                outputData.poseResponse.data = outputPoseResponse;
            }
        })
        .catch(err => console.error("Frontend TF: Error in pose prediction: ", err))
}

function poseLearn(data) {
    try {
        const poseEmotion = tf.tensor1d(data.flat());
        poseClassifier.addExample(poseEmotion, poseClIndex);
        console.log("Learnt new pose with index: ", poseClIndex)
        poseClIndex++;      //firstly increment the pose classifier index
    }
    catch (err) {
        console.error("Error in pose learn: ", err.message)
    }
}

//verbal classifier for the backend
function restoreVerbalClassifier(data) {
    var tensorObj = data;
    Object.keys(tensorObj).forEach((key) => {
        tensorObj[key] = tf.tensor(tensorObj[key], [1, tensorObj[key].length]);
    })
    console.log("Frontend TF: Verbal classifier restored.");
    verbalClassifier.setClassifierDataset(tensorObj);
    verbalClIndex = verbalClassifier.getNumClasses();       //get the next-to-last index of the verbal classifier
    if (verbalClIndex == undefined) {
        console.error("Verbal classifier index is undefined in restoreVerbalClassifier().");
        verbalClIndex = 0;
    }
}

let displayedOnceVerbalAns = false;
function predictVerbalAnswers(Indata) {
    if (verbalClIndex !== undefined && verbalClIndex !== 0) {
        const verbalData = tf.tensor1d(Indata.flat());
        verbalClassifier.predictClass(verbalData)
            .then(classId => {
                if (outputData.verbalResponse.label !== null) {
                    setTimeout(function () {
                        outputData.verbalResponse.label = classId.label;
                        outputData.verbalResponse.score = classId.confidences[classId.label];
                    }, outputCommDelay)
                }
                else {
                    outputData.verbalResponse.label = classId.label;
                    outputData.verbalResponse.score = classId.confidences[classId.label];
                }
            })
            .catch(err => {
                console.error("Frontend TF: Error in verbal answer prediction: ", err);
            })
    }
    else if (displayedOnceVerbalAns == false) {
        console.warn("Verbal classifier cannot predict because it is not restored.");
        displayedOnceVerbalAns = true;
    }
}

//here data shall contain the emotion, topic, and some other features
function VerbalLearn(data) {
    try {
        if (verbalClIndex == undefined) {
            console.error("verbalClIndex is undefined in verbalLearn().");
            verbalClIndex = 0;
        }
        const verbalTensor = tf.tensor1d(data.flat());
        verbalClassifier.addExample(verbalTensor, verbalClIndex);
        console.log("Learnt new verbal response with index: ", verbalClIndex);
        verbalClIndex++;        //firstly increment the verbal index and then learn the new verbal response
    }
    catch (err) {
        console.error("Error in verbal learn: ", err.message)
    }
}

//save the verbal classifier
function saveVerbalAnswers(type) {
    if (type == undefined)
        type = 'saveFinal';     //otherwise savePeriodic
    let datasetObj = {};
    //if the classifier has been restored
    if (verbalClIndex !== undefined && verbalClIndex != 0) {
        var dataset = verbalClassifier.getClassifierDataset();
        Object.keys(dataset).forEach((key) => {
            let data = dataset[key].dataSync();
            datasetObj[key] = Array.from(data);
        });
    }
    else {
        datasetObj = null;
    }

    let dataToSave = {
        verbalSave: {
            data: datasetObj,
            type: type
        }
    }
    fetch('http://127.0.0.1:5000/save', {
        headers: jsonConf,
        body: JSON.stringify(dataToSave),
        method: 'POST'
    })
        .then(() => {
            console.log("Frontend TF: Verbal classifier save dataset sent.");
        })
        .catch(err => console.error('Frontend TF: Error sending verbal save object to backend: ', err))
}

//Pose detection analysis
//order of the parts in the array------
//nose                  //leftHip
//leftEye               //rightHip
//rightEye              //leftKnee
//leftEar               //rightKnee
//rightEar              //leftAnkle
//leftShoulder          //rightAnkle
//rightShoulder         
//leftElbow
//rightElbow
//leftWrist
//rightWrist
//pose detection tensorflow variables as it has been transferred to frontend
//this function takes in the pose detection response and converts it into appropriate servo degrees
let RightUpperArmMaxLength;
let LeftUpperArmMaxLength;
let RightLowerArmMaxLength;
let LeftLowerArmMaxLength;
let maxHorHeadLength;
let maxHorBodyLength;
let maxBodyDist;
function analyze_poseDetection(positions) {
    var detectedPositionsX = new Array(0);
    var detectedPositionsY = new Array(0);
    for (var x = 0; x < positions.keypoints.length; x++) {
        detectedPositionsX[x] = positions.keypoints[x].position.x;
        detectedPositionsY[x] = positions.keypoints[x].position.y;
    }
    //here we have three arrays containing info about the details
    //code to extract the distances from the poses detected

    //get the maximum distance of the body (in the vertical direction)
    //midpoint of left shoulder and right shoulder
    var upperMidpointX = round((detectedPositionsX[5] + detectedPositionsX[6]) / 2);
    var upperMidpointY = round((detectedPositionsY[5] + detectedPositionsY[6]) / 2);
    //midpoint of the left ankle and right ankle
    var lowerMidpointX = round((detectedPositionsX[15] + detectedPositionsX[16]) / 2);
    var lowerMidpointY = round((detectedPositionsY[15] + detectedPositionsY[16]) / 2);
    //get the cartesian distance of the body (However the scale here is not constant and might hamper)
    var thismaxBodyDist = Math.sqrt(pow(upperMidpointX - lowerMidpointX, 2) + pow(upperMidpointY - lowerMidpointY, 2));
    if (maxBodyDist == undefined || thismaxBodyDist > maxBodyDist)
        maxBodyDist = thismaxBodyDist;
    //from this body distance we might be able to get the size of the body and then scale our degrees based on this size
    //get the degrees my measuring the shoulder and elbow length
    //Cartesian distance between the elbow and shoulder servos of respective arms
    var leftShoulder_leftElbow = Math.sqrt(pow(detectedPositionsX[5] - detectedPositionsX[7], 2) + pow(detectedPositionsY[5] - detectedPositionsY[7], 2));
    var rightShoulder_rightElbow = Math.sqrt(pow(detectedPositionsX[6] - detectedPositionsX[8], 2) + pow(detectedPositionsY[6] - detectedPositionsY[8], 2));
    //from this we are now getting the actual servo degrees
    var leftShoulderYServoUpperArm;
    var rightShoulderYServoUpperArm;

    //make sure we don't exceed boundaries
    if (LeftUpperArmMaxLength == undefined || leftShoulder_leftElbow > LeftUpperArmMaxLength) {
        //the arm length cannot exceed the vertical body distance
        if (leftShoulder_leftElbow > maxBodyDist) {
            LeftUpperArmMaxLength = maxBodyDist;
            leftShoulder_leftElbow = LeftUpperArmMaxLength;
        }
        else
            LeftUpperArmMaxLength = leftShoulder_leftElbow;
    }
    if (RightUpperArmMaxLength == undefined || rightShoulder_rightElbow > RightUpperArmMaxLength) {
        if (rightShoulder_rightElbow > maxBodyDist) {
            RightUpperArmMaxLength = maxBodyDist;
            rightShoulder_rightElbow = RightUpperArmMaxLength;
        }
        else
            RightUpperArmMaxLength = rightShoulder_rightElbow;
    }

    //here we get the actual servo degrees for the shoulder servos
    if (detectedPositionsY[5] - detectedPositionsY[7] < 0) { //left elbow is higher than the left shoulder
        //left hand is moving up, left shoulder servo also will be moving up
        leftShoulderYServoUpperArm = map(leftShoulder_leftElbow, 0, LeftUpperArmMaxLength, 90, 180);
    }
    else {
        //left hand is moving down, left shoulder servo also will be moving down
        leftShoulderYServoUpperArm = map(leftShoulder_leftElbow, 0, LeftUpperArmMaxLength, 90, 0);
    }

    if (detectedPositionsY[6] - detectedPositionsY[8] < 0) { //right elbow is higher than the right shoulder
        //right hand is moving up
        rightShoulderYServoUpperArm = map(rightShoulder_rightElbow, 0, RightUpperArmMaxLength, 90, 180);
    }
    else {
        //hand is moving down
        rightShoulderYServoUpperArm = map(rightShoulder_rightElbow, 0, RightUpperArmMaxLength, 90, 0);
    }

    //get the degrees by measuring the length of the elbow and the wrist
    let leftLowerArmlength = Math.sqrt(pow(detectedPositionsX[7] - detectedPositionsX[9], 2) + pow(detectedPositionsY[7] - detectedPositionsY[9], 2));
    let rightLowerArmlength = Math.sqrt(pow(detectedPositionsX[8] - detectedPositionsX[10], 2) + pow(detectedPositionsY[8] - detectedPositionsY[10], 2));
    //no boundaries exceed
    if (LeftLowerArmMaxLength == undefined || LeftLowerArmMaxLength < leftLowerArmlength) {
        if (leftLowerArmlength > maxBodyDist) {
            LeftLowerArmMaxLength = maxBodyDist;
            leftLowerArmlength = LeftLowerArmMaxLength;
        }
        else
            LeftLowerArmMaxLength = leftLowerArmlength;
    }
    if (RightLowerArmMaxLength == undefined || RightLowerArmMaxLength < rightLowerArmlength) {
        if (rightLowerArmlength > maxBodyDist) {
            RightLowerArmMaxLength = maxBodyDist;
            rightLowerArmlength = RightLowerArmMaxLength;
        }
        else
            RightLowerArmMaxLength = rightLowerArmlength;
    }
    //the output servo degrees
    var leftShoulderYServo = round(leftShoulderYServoUpperArm);
    var rightShoulderYServo = round(rightShoulderYServoUpperArm);

    var leftShoulderYServoLowerArm;
    var rightShoulderYServoLowerArm;
    //for the left arm
    if (detectedPositionsY[7] - detectedPositionsY[9] < 0) {
        //lower arm is moving up, lower arm servo will also move up
        leftShoulderYServoLowerArm = round(map(leftLowerArmlength, 0, LeftLowerArmMaxLength, 90, 180));
    }
    else {
        //lower arm moving down, lower arm servo will also move down
        leftShoulderYServoLowerArm = round(map(leftLowerArmlength, 0, LeftLowerArmMaxLength, 90, 0));
    }

    //the right arm
    if (detectedPositionsY[8] - detectedPositionsY[10] < 0) {
        //hand is moving up
        rightShoulderYServoLowerArm = round(map(rightLowerArmlength, 0, RightLowerArmMaxLength, 90, 180));
    }
    else {
        //hand is moving down
        rightShoulderYServoLowerArm = round(map(rightLowerArmlength, 0, RightLowerArmMaxLength, 90, 0));
    }
    //still haven't included the elbow servo degrees in the output
    var leftElbowServo = round(leftShoulderYServoLowerArm);
    var rightElbowServo = round(rightShoulderYServoLowerArm);
    //--------------------------------Horizontal shoulder servo-------------------------------------
    var posDegreesLeft = round(Math.abs(angleWithYaxis(detectedPositionsX[5], detectedPositionsY[5], detectedPositionsX[7], detectedPositionsY[7])));
    var posDegreesRight = round(Math.abs(angleWithYaxis(detectedPositionsX[6], detectedPositionsY[6], detectedPositionsX[8], detectedPositionsY[8])));
    //horizontal servo degrees
    //right hand servo segrees are opposite of the left hand servo degrees
    var leftShoulderXServo = posDegreesLeft;
    var rightShoulderXServo = posDegreesRight;

    //--------------------------------Wrist Servo----------------------------------------------------
    let leftWristDegrees = round(Math.abs(angleWithYaxis(detectedPositionsX[7], detectedPositionsY[7], detectedPositionsX[9], detectedPositionsY[9])));
    let rightWristDegrees = round(Math.abs(angleWithYaxis(detectedPositionsX[8], detectedPositionsY[8], detectedPositionsX[10], detectedPositionsY[10])));
    //in the current prototype version of the program, we are not using the stabilizer although it is a very useful algorithm
    //elbow servo degrees
    var leftWristServo = leftWristDegrees;
    var rightWristServo = rightWristDegrees;

    //----------------------------body movement including all---------------------------------------------------
    var leftShoulder_leftHip = Math.sqrt(pow(detectedPositionsX[5] - detectedPositionsX[11], 2) + pow(detectedPositionsY[5] - detectedPositionsY[11], 2));
    var rightShoulder_rightHip = Math.sqrt(pow(detectedPositionsX[6] - detectedPositionsX[12], 2) + pow(detectedPositionsY[6] - detectedPositionsY[12], 2));
    var leftShoulder_rightShoulder = Math.sqrt(pow(detectedPositionsX[5] - detectedPositionsX[6], 2) + pow(detectedPositionsY[5] - detectedPositionsY[6], 2));
    //we then determine the side the human is looking based upon perspective (the farther side has the smaller distance)
    if (maxHorBodyLength == undefined || leftShoulder_rightShoulder > maxHorBodyLength) {
        if (leftShoulder_rightShoulder > maxBodyDist) {
            maxHorBodyLength = maxBodyDist;
            leftShoulder_rightShoulder = maxHorBodyLength;
        }
        else
            maxHorBodyLength = leftShoulder_rightShoulder;
    }
    let baseServoDegrees;
    if (leftShoulder_leftHip > rightShoulder_rightHip)
        baseServoDegrees = round(map(leftShoulder_rightShoulder, maxHorBodyLength, 0, 90, 0));    //human is looking right
    else baseServoDegrees = round(map(leftShoulder_rightShoulder, maxHorBodyLength, 0, 90, 180)); //human is looking left

    //--------------------------head movement only------------------------------------------------
    var leftEar_Nose = Math.sqrt(pow(detectedPositionsX[0] - detectedPositionsX[3], 2) + pow(detectedPositionsY[0] - detectedPositionsY[3], 2));
    var rightEar_Nose = Math.sqrt(pow(detectedPositionsX[0] - detectedPositionsX[4], 2) + pow(detectedPositionsY[0] - detectedPositionsY[4], 2));
    var leftEar_rightEar = Math.sqrt(pow(detectedPositionsX[3] - detectedPositionsX[4], 2) + pow(detectedPositionsY[3] - detectedPositionsY[4], 2));
    if (maxHorHeadLength == undefined || maxHorHeadLength < leftEar_rightEar) {
        if (leftEar_rightEar > maxBodyDist) {
            maxHorHeadLength = maxBodyDist;
            leftEar_rightEar = maxHorHeadLength;
        }
        else
            maxHorHeadLength = leftEar_rightEar;
    }
    let headServoHorDegree;
    if (leftEar_Nose > rightEar_Nose) {
        //human is looking right
        headServoHorDegree = round(map(leftEar_rightEar, maxHorHeadLength, 0, 90, 0));
    }
    else {
        //human is looking left
        headServoHorDegree = round(map(leftEar_rightEar, maxHorHeadLength, 0, 90, 180));
    }

    return [headServoHorDegree, 90, rightShoulderYServo, leftShoulderYServo, rightShoulderXServo, leftShoulderXServo,
        rightWristServo, leftWristServo, baseServoDegrees, 90, 90, rightElbowServo, leftElbowServo];
    //return the poses with respect to the connections and the series in the Arduino Code
    //head horizontal, head vertical, right shoulder vertical ,left shoulder vertical ,right shoulder horizontal
    //left shoulder horizontal, right elbow, left elbow, base ,right ear,left ear,right elbow, left elbow
    //function to map the data to the given range
    function angleWithYaxis(x1, y1, x2, y2) {
        //calculates the angle in non-negative degreees of line joining two points with the Y axis
        const m1 = (y2 - y1) / (x2 - x1);
        const m2 = 0;                                       //for the x axis
        let tangent = Math.abs((m1 - m2) / (1 + m1 * m2));
        let atan = (Math.atan(tangent) * 180 / 3.14) + 90;  //positive degree measure [0 to 180 degrees]
        return (atan);
    }
    function map(x, in_min, in_max, out_min, out_max) {
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
    }
    //we round off the given function, I know i am being stupid doing it this way
    function round(el) {
        return Math.round(el)
    }
    function pow(el, power) {
        return Math.pow(el, power)
    }
}

//function to send all the output at once, 
async function sendOutput() {
    while (true) {
        await fetch('http://127.0.0.1:5000/output', {
            headers: jsonConf,
            method: 'POST',
            body: JSON.stringify(outputData)
        })
            .then(() => {
                //clear the responses already sent, replacing with null as undefined is invalid in JSON
                if (outputData.poseDetect.data !== null) {
                    outputData.poseDetect.data = null;
                    outputData.poseDetect.faceTracker.data = null;
                }
                if (outputData.objectDetect.data !== null)
                    outputData.objectDetect.data = null;
                if (outputData.faceAPI.data !== null)
                    outputData.faceAPI.data = null;
                if (outputData.textToxicity.data !== null)
                    outputData.textToxicity.data = null;
                if (outputData.visualObjectsCustom.data !== null)
                    outputData.visualObjectsCustom.data = null;
                if (outputData.audioObjects.data !== null)
                    outputData.audioObjects.data = null;
                if (outputData.speakerResponse.data !== null)
                    outputData.speakerResponse.data = null;
                if (outputData.songResponse.data !== null)
                    outputData.songResponse.data = null;
                if (outputData.poseResponse.data !== null) {
                    outputData.poseResponse.data = null;
                    outputData.poseResponse.type = null;
                }
                if (outputData.verbalResponse.label !== null) {
                    console.log("Frontend TF: verbal prediction label: ", outputData.verbalResponse.label)
                    outputData.verbalResponse.label = null;
                    outputData.verbalResponse.score = null;
                }
                if (outputData.recognisedSpeech.data !== null) {
                    console.log("Frontend TF: recognised speech in object: ", outputData.recognisedSpeech.data);
                    outputData.recognisedSpeech.data = null;
                }
                if (outputData.songConfirmation.data !== null)
                    outputData.songConfirmation.data = null;
                if (outputData.askSingingConfirmation.data !== null)
                    outputData.askSingingConfirmation.data = null;
                if (outputData.songOutgoingData.data !== null)
                    outputData.songOutgoingData.data = null;
            })
            .catch(err => console.error('Frontend TF: Error sending detections to backend: ', err))
        await sleep(outputCommDelay);          //delay after every sending function
    }
}

function loadClient() {
    gapi.client.setApiKey(CSE_apikey);
    return gapi.client.load("https://content.googleapis.com/discovery/v1/apis/customsearch/v1/rest")
        .then(function () { console.log("GAPI client loaded for API"); },
            function (err) { console.error("Error loading GAPI client for API", err); });
}

let cseImage = document.getElementById('cseImage');
cseImage.style.visibility = 'hidden';

async function visualLearnStimulus(label, classId) {
    //this function returns an array of images 
    const options = {
        cx: CSE_id,                     //the custom search engine id: 
        num: 10,                        //number of search results (1-10 inclusive)
        q: label,                       //the search query
        searchType: "image",            //the type of search ("image" for our case)
        //imgType: "clipart", "face", "lineart", "stock", "photo", "animated"
        //imgSize: "huge", "icon", "large", "medium", "small", "xlarge", "xxlarge"
        //imgColorType: "color", "gray", "mono": black and white, "trans": transparent background
        //imgDominantColor: "black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red", "teal", "white", "yellow"
    }
    gapi.client.search.cse.list(options)
        .then(function (response) {
            response.result.items.forEach(el => {
                //change the image for every picture
                cseImage.src = el.image.thumbnailLink;
                cseImage.width = "250";
                cseImage.height = "200";
                const activation = visual_net.infer(cseImage, 'conv_preds');
                visual_classifier.addExample(activation, classId);
                console.log("Stimulus Inclined Visual image learnt for ", label)
            })
            visual_recognition_busy = false;
        },
            function (err) {
                console.error("Custom Search engine error: ", err);
                visual_recognition_busy = false;
            });
}