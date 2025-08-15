// @ts-check
//Javascript frontend
//Robotic Intuition Operator
//Debashish Buragohain
//in the latest version the main frontend includes all the compoments in a single file
"use-strict"
var startFrontend = false;          //global switch whether to start the frontend or not
const testMode = true;              //if RIO is implemented in the test mode
if (testMode) {
    console.warn("RIO is being implemented in test mode.");
}
//the button is hidden from the start
var startBtn = document.getElementsByClassName("startBtn")[0];
//variables to be shared with the frontend javascript file
const defaultPitch = "1.1";
const defaultSpeed = "0.9";
var responseText = "";   //setting the default response text to fire up the speech synthesis
var current_emotion = 'neutral';
var rate = defaultPitch;            //defaulting to these values
var pitch = defaultSpeed;
var singingState;                  //if the robot is singing, from the backend
var songIncomingMessage = "";
//the recognised speech variables are shared with frontend tesorflow
var final_transcript = "";
var interim_transcript = "";
let askedSongConfirmation = false;
var stopListening = false;
var ContinueListening = false;
//a single unused JSON output object, shared with the frontend tensorflow file
var outputData = {
    poseDetect: {       //pose detection output
        data: null,
        faceTracker: {          //face tracking nose position, a part of the pose detection results
            data: null
        }
    },
    objectDetect: {
        data: null
    },
    faceAPI: {
        data: null
    },
    textToxicity: {
        data: null
    },
    visualObjectsCustom: {
        data: null
    },
    audioObjects: {
        data: null
    },
    speakerResponse: {
        data: null
    },
    songResponse: {
        data: null
    },
    poseResponse: {         //the pose classifier output
        data: null,
        type: null
    },
    verbalResponse: {       //the verbal classifier output
        label: null,
        score: null
    },
    recognisedSpeech: {     //the speech recognition property for the frontend javascript
        data: null
    },
    songConfirmation: {     //song confirmation property for the frontend javascript
        data: null
    },
    askSingingConfirmation: {//another song confirmation property for frontend javascript
        data: null
    },
    songOutgoingData: {     //the outgoing data for frontend javascript
        data: null
    }
}
const minMicVol = 10;                  //enter the minimum volume of the mic input
const maxListenInterval = 6000;        //the maximum time to wait in case of a silence period
//initialised to 3000 at first
const gap = 125;                      //time in miliseconds between each note while listening to the song
//we might need to increase the gaps as well
//16th in one second
var detectedNote = null;             //detected note defaulted to null
var micVol = 0;                        //mic volume defaulted to 0
var song = new Array(0);               //the song array contains all the notes serially to the end
var last_emotion = 'neutral';          //defaulted to the neutral emotion
var synth = window.speechSynthesis;
var voices = [];
var ifSpoken = false;                  //if we have already spoken in the current iteration
//when ready, change the interval to a low value
var blinked = false;                   //initially the robot does not blink
const blink_time = 400;                //the blinking time in milis
//the variables controlling the singing of the robot
var singing = false;
var singing_fromBefore = false;
//the global variables defining the address of the eyes
var image_url_emotion = './facial_expressions/animatedEyes/neutralEyesGIF.gif';
var image_url_speech = './facial_expressions/animatedNoSpeeches/neutral_notSpeakGIF.gif';
var frontendJSready = false;                //global variable determining if the frontend JS is ready
const initialisationDelayJS = 500;          //this is just the sleeping time, no memory usage
const analyserOptions = {
    callback: analyserCallback,
    returnNote: true,
    returnCents: false,
    decimals: 2
}

const analyser = new PitchAnalyser(analyserOptions);
//initialise the speech synthesis
if (speechSynthesis.onvoiceschanged != undefined) {
    speechSynthesis.onvoiceschanged = populateVoiceList;
}
const main_interval = 1000;         //set to 1000 at first
const sleep = ms => new Promise(req => setTimeout(req, ms));
//check if the internet is connected or not
var ifConnectedJS = window.navigator.onLine;
if (ifConnectedJS == true) initJSFrontend();
else console.error('Frontend JS: Internet connection not available. Please connect to the internet and refresh.');

async function initJSFrontend() {
    //the javascript frontend is ready at this point
    frontendJSready = true
    //wait until all the modules are ready
    while (startFrontend == false)
        await sleep(initialisationDelayJS);
    //initialise the screen by displaying the neutral emotion
    console.log('Frontend JS: Initialising the note analyser..')
    analyser.initAnalyser()
        .then(() => {
            console.log('Frontend JS: Note Analyser successfully loaded.');
            function initFunctionsExpressions() {
                populateVoiceList();
                modifiedHTML_emotion();
                modifiedHTML_speech();
            }
            initFunctionsExpressions();
            //initialisation speech
            speakout("I am Rio., , And now, I am finally alive.", true, defaultPitch, defaultSpeed);
            startJS();
        })
        .catch(err => console.error('Frontend JS: Error in starting the note analyser: ', err))
}
//---------------------------------------Functions Block------------------------------------------------------
function populateVoiceList() {
    voices = synth.getVoices();                 //get the voices available
    console.log("Frontend JS: Voices available: ", JSON.stringify(voices))
}

var recognizing,
    transcription = document.getElementById('speech'),
    interim_span = document.getElementById('interim');
var speech = new webkitSpeechRecognition() || speechRecognition();

function startJS() {
    analyser.audioContext.resume();             //we are not stopping the analyser after starting once
    analyser.startAnalyser();
    async function getMicVol() {
        //get the volume of the microphone
        //please keep in mind that this function will loop automatically and we don't need to put any loop here
        navigator.mediaDevices.getUserMedia({ audio: true, video: false })
            .then(function (stream) {
                let audioContext = new AudioContext();
                let Volanalyser = audioContext.createAnalyser();
                let microphone = audioContext.createMediaStreamSource(stream);
                let javascriptNode = audioContext.createScriptProcessor(2048, 1, 1);
                Volanalyser.smoothingTimeConstant = 0.8;
                Volanalyser.fftSize = 1024;
                microphone.connect(Volanalyser);
                Volanalyser.connect(javascriptNode);
                javascriptNode.connect(audioContext.destination);
                javascriptNode.onaudioprocess = function () {
                    var array = new Uint8Array(Volanalyser.frequencyBinCount);
                    Volanalyser.getByteFrequencyData(array);
                    var values = 0;
                    var length = array.length;
                    for (var i = 0; i < length; i++) {
                        values += (array[i]);
                    }
                    micVol = Math.round(values / length);
                    //return average;
                }
            })
            .catch(function (err) {
                console.error('Frontend JS: Error reading the mic volume: ' + err);
            });
    }
    getMicVol();
    //speech recognition halted in the test mode
    if (testMode == false) {
        //Speech recognition block
        //we don't need any onload listener for speech recognition as this will have already loaded by the time it is called
        if (!(window.webkitSpeechRecognition) && !(window.speechRecognition))
            console.log('Frontend JS: Please use Google Chrome for the best experience.');
        else {
            function reset() {
                recognizing = false;
                interim_span.innerHTML = '';
                transcription.innerHTML = '';
                speech.start();
            }
            interim_span.style.opacity = '0.5';
            speech.continuous = true;
            speech.interimResults = true;
            speech.lang = 'en-US'; // check google web speech example source for more lanuages
            speech.start();        //enables recognition on default
            speech.onstart = function () {
                // When recognition begins
                console.log('Frontend JS: Started recognizing speech again.')
                recognizing = true;
            };
            //speech recognition has produced a result
            speech.onresult = function (event) {
                interim_transcript = '';
                final_transcript = '';
                // main for loop for final and interim results
                for (var i = event.resultIndex; i < event.results.length; ++i) {
                    if (event.results[i].isFinal) {
                        final_transcript += event.results[i][0].transcript;
                    } else {
                        interim_transcript += event.results[i][0].transcript;
                    }
                }
                transcription.innerHTML = final_transcript;
                interim_span.innerHTML = interim_transcript;
                //send the final recognised speech to the backend
                if (final_transcript.length != 0) {
                    let sendText = final_transcript.slice(0);
                    //if the robot is asking for the song confirmation send as song confirmation command
                    if (askedSongConfirmation == true) {
                        let typeOfSpeech = checkYesOrNo(sendText.toLowerCase());
                        if (typeOfSpeech == 'yes_word') {
                            outputData.songConfirmation.data = 'yes';
                            askedSongConfirmation = false;
                        }
                        else if (typeOfSpeech == 'no_word') {
                            outputData.songConfirmation.data = 'no';
                            askedSongConfirmation = false;
                        }
                        else if (typeOfSpeech == 'none_') {
                            outputData.recognisedSpeech.data = sendText;
                        }
                        function checkYesOrNo(givenText) {
                            let yesWords = ['yes', 'sure', 'affirmative', 'amen', 'fine', 'good', 'okay', 'true', 'yea', 'all right', 'aye', 'beyond a doubt', 'by all means', 'certainly', 'definitely', 'even so',
                                'exactly', 'gladly', 'good enough', 'granted', 'indubitably', 'just so', 'most assuredly', 'naturally', 'of course', 'positively', 'precisely', 'sure thing', 'surely',
                                'undoubtedly', 'unquestionably', 'very well', 'willingly', 'without fail', 'yep'];

                            for (var i = 0; i < yesWords.length; i++) {
                                if (givenText.includes(yesWords[i])) {
                                    return "yes_word";
                                }
                            }

                            let noWords = ["don't", "no", "nay", "nix", "never", "not"]
                            for (var i = 0; i < noWords.length; i++) {
                                if (givenText.includes(noWords[i])) {
                                    return "no_word";
                                }
                            }
                            return "none_";
                        }
                    }
                    //otherwise this is a general speech
                    else outputData.recognisedSpeech.data = sendText;
                }
            };
            speech.onerror = function (event) {
                // Either 'No-speech' or 'Network connection error', ignore these errors
                console.error(`error in speech recognition: ${event.error}`);
            };
            speech.onend = function () {
                // When recognition ends
                reset();
            };
        }
    }

    //the emotion and the speech response function
    async function mainJS() {
        while (true) {
            //general communcations module
            if (responseText.length !== 0) console.log('Frontend JS: Response text from backend: ', responseText);
            //current_emotion already updated in the frontend TF            
            if (current_emotion == null || current_emotion == "" || current_emotion == undefined)
                current_emotion = last_emotion;

            //singingState is updated in the frontend TF
            if (singingState !== null) {
                if (singingState == true) {
                    singing = true;
                    //if we are singing for the first iteration
                    //only then change the singing state
                    if (singing_fromBefore == false) {
                        singing_fromBefore = true;
                        console.log('Frontend JS: Singing has started');
                        determineSpeechImgURL();
                        modifiedHTML_speech();
                    }
                }
                else if (singing == true) {
                    singing = false;
                    singing_fromBefore = singing;
                    console.log('Frontend JS: Singing has stopped.');
                    determineNoSpeechImgURL();
                    modifiedHTML_speech();
                }
            }

            //if we are not singing
            if (singing == false) {
                //if we have not already spoken
                if (ifSpoken == false) {
                    if (responseText.length !== 0) {
                        if (responseText.toLowerCase().includes('should i save the song that i have just learnt?     ')) {
                            askedSongConfirmation = true;
                        }
                        speakout();
                    }
                }
                //else we skip speaking for the current iteration
                else ifSpoken = true;
            }

            changeEmotionState();               //update the emotion state in every iteration
            //singing communication halted in the test mode
            if (testMode == false) {
                if (songIncomingMessage.length !== 0) {
                    switch (songIncomingMessage) {
                        //if the backend asks us to listen to the song
                        case "listen":
                            console.log('Frontend JS: Listening and learning new song.');
                            listenToSong();
                            break;
                        case "stopListening":
                            console.log("Stopping to listen to the song.");
                            if (ContinueListening == true)
                                stopListening = true;
                            else console.warn("Frontend JS: Cannot stop listening when not started to listen.");
                            break;
                        //when the backend has confirmed the song learnt
                        case "yes":
                            console.log('Frontend JS: Recently learnt song sent to the backend.')
                            sendSongToBackend(song);
                            console.log("Frontend JS: Song saved sucessfully.");
                            //the song is stored by default
                            break;
                        case "no":
                            console.log("Frontend JS: Song save aborted.");
                            song.length = 0;    //clear the learnt song
                            break;
                    }
                    songIncomingMessage = "";       //clear the song incoming message after every sending function
                }
            }
            await sleep(main_interval);
        }
    }
    mainJS();
}

function changeEmotionState() {
    //blink and change the emotion only when the current emotion is not equal to the previous emotion
    if (current_emotion != last_emotion) {
        if (blinked == false) {
            //blink first and then change the emotion
            image_url_emotion = "./facial_expressions/animatedEyes/blinkFinalPNG.png";
            modifiedHTML_emotion();
            blinked = true;             //we have just blinked
            //wait for some time for the blink to be prominent
            //determine the correct emotion for the robot
            determineEmotionURL();
            setTimeout(function () {
                //time to display the new emotion
                modifiedHTML_emotion();
                blinked = false;
            }, blink_time);
        }
        last_emotion = current_emotion;     //update the variable
    }
}

//add the animations
function modifiedHTML_emotion() {
    console.log("Image url emotion: ", image_url_emotion);
    document.querySelector('#emotion_area').style.background = 'url("' + image_url_emotion + '") no-repeat center center';
}

function modifiedHTML_speech() {
    console.log('Image url speech: ', image_url_speech);
    document.querySelector('#speech_area').style.background = 'url("' + image_url_speech + '") no-repeat center ';
}

//the speaking function with the Web Speech API
function speakout(data, Ifdefault, givenPitch, givenSpeed, givenVoiceIndex) {
    var speakText;          //the text to be spoken
    if (data == undefined) {
        //the responseText will be set to null after we have initiated the speakout function
        speakText = new SpeechSynthesisUtterance(responseText);
        responseText = "";
    }
    else speakText = new SpeechSynthesisUtterance(data);
    //if we are given a specific voice and not the default one
    if (Ifdefault !== undefined && Ifdefault !== null && Ifdefault !== true) {
        speakText.voice = voices[givenVoiceIndex];
    }
    speakText.pitch = (givenPitch !== undefined) ? givenPitch : pitch;
    speakText.rate = (givenSpeed !== undefined) ? givenSpeed : rate;
    //speak our thing and also dislpay the emotions
    synth.speak(speakText);
    //dislpay the emotions after some time interval for being more prominent

    setTimeout(async () => {
        determineSpeechImgURL();
        modifiedHTML_speech();
    }, 800);

    speakText.onpause = function (event) {
        var char = event.utterance.text.charAt(event.charIndex);
        console.log('Frontend JS: Speech paused at character ' + event.charIndex + ' of "' +
            event.utterance.text + '", which is "' + char + '".');
        determineNoSpeechImgURL();
        modifiedHTML_speech();
    }
    speakText.onend = e => {
        determineNoSpeechImgURL();
        modifiedHTML_speech();
    }
}

function determineEmotionURL() {
    switch (current_emotion) {
        //set the url of the expression image file based on the current emotion of the system
        case "joy":
            image_url_emotion = './facial_expressions/animatedEyes/happyEyesSlightGIF.gif';
            break;
        case "joy_very":
            image_url_emotion = './facial_expressions/animatedEyes/happyEyesVeryGIF.gif';
            break;
        case "sadness":
            image_url_emotion = './facial_expressions/animatedEyes/sadEyesSlightGIF.gif';
            break;
        case "sadness_very":
            image_url_emotion = './facial_expressions/animatedEyes/sadEyesVeryGIF.gif';
            break;
        case "anger":
            image_url_emotion = './facial_expressions/animatedEyes/angryEyesSlightGIF.gif';
            break;
        case "anger_very":
            image_url_emotion = './facial_expressions/animatedEyes/angryEyesVeryGIF.gif';
            break;
        case "fear":
            image_url_emotion = './facial_expressions/animatedEyes/afraidEyesSlightGIF.gif';
            break;
        case "fear_very":
            image_url_emotion = './facial_expressions/animatedEyes/afraidEyesVeryGIF.gif';
            break;
        case 'surprise':
            image_url_emotion = './facial_expressions/animatedEyes/surprisedEyesSlightGIF.gif';
            break;
        case 'surprise_very':
            image_url_emotion = './facial_expressions/animatedEyes/surprisedEyesVeryGIF.gif';
            break;
        case 'disgust':
            image_url_emotion = './facial_expressions/animatedEyes/disgustedEyesSlightGIF.gif';
            break;
        case 'disgust_very':
            image_url_emotion = './facial_expressions/animatedEyes/disgustedEyesVeryGIF.gif';
            break;
        default:
            //neutral is always the default one
            image_url_emotion = './facial_expressions/animatedEyes/neutralEyesGIF.gif';
            break;
    }
}
//determines the speech URL while the robot is speaking 
function determineSpeechImgURL() {
    switch (current_emotion) {
        case "joy":
        case "joy_very":
            image_url_speech = './facial_expressions/animatedSpeeches/happySpeakingRAW.gif';
            break;
        case "sadness":
        case "sadness_very":
            image_url_speech = './facial_expressions/animatedSpeeches/sadSpeakingRAW.gif';
            break;
        case "fear":
        case "fear_very":
            image_url_speech = './facial_expressions/animatedSpeeches/afraid_angrySpeakingRAW.gif';
            break;
        case "anger":
        case "anger_very":
            image_url_speech = './facial_expressions/animatedSpeeches/afraid_angrySpeakingRAW.gif';
            break;
        case "surprise":
        case "surprise_very":
            image_url_speech = './facial_expressions/animatedSpeeches/surprised_disgustSpeakingRAW.gif';
            break;
        case "disgust":
        case "disgust_very":
            image_url_speech = './facial_expressions/animatedSpeeches/surprised_disgustSpeakingRAW.gif';
            break;
        default:
            //neutral emotion is the default one
            image_url_speech = './facial_expressions/animatedSpeeches/neutralSpeakingRAW.gif';
            break;
    }
}
//determines the speech URL when the robot is not speaking
function determineNoSpeechImgURL() {
    switch (current_emotion) {
        case "joy":
        case "joy_very":
            image_url_speech = './facial_expressions/animatedNoSpeeches/happy_notSpeakGIF.gif';
            break;
        case "sadness":
        case "sadness_very":
            image_url_speech = './facial_expressions/animatedNoSpeeches/sad_notSpeakGIF.gif';
            break;
        case "fear":
        case "fear_very":
            image_url_speech = './facial_expressions/animatedNoSpeeches/angry_afraid_notSpeakGIF.gif';
            break;
        case "anger":
        case "anger_very":
            image_url_speech = './facial_expressions/animatedNoSpeeches/angry_afraid_notSpeakGIF.gif';
            break;
        case "surprise":
        case "surprise_very":
            image_url_speech = './facial_expressions/animatedNoSpeeches/surprised_disgust_notSpeakGIF.gif';
            break;
        case "disgust":
        case "disgust_very":
            image_url_speech = './facial_expressions/animatedNoSpeeches/surprised_disgust_notSpeakGIF.gif';
            break;
        default:
            //neutral emotion is the default one
            image_url_speech = './facial_expressions/animatedNoSpeeches/neutral_notSpeakGIF.gif';
            break;
    }
}

async function listenToSong() {
    alert("Started listening.");
    song.length = 0;
    var endTime = 0;                        //the end of the time period, initialized to 0    
    var startTime = new Date().getTime();   //the start of the period of listening to the song
    ContinueListening = true;               //set this to true for listening to the song
    while (ContinueListening == true) {
        //make sure everything is proper for listening to the song
        if (micVol >= minMicVol && typeof detectedNote == 'string') {
            console.log("detected note pushed into song array: ", detectedNote)
            song.push(detectedNote)
        }
        else {
            if (endTime - startTime <= maxListenInterval) {
                endTime = new Date().getTime();
                console.log("Null note pushed into song array.");
                song.push('null');              //in this case the 'null' is a string which represents a silence period
            }
            else if (endTime - startTime > maxListenInterval || stopListening == true) {
                //end the listening and ask for confirmation to the user
                console.log('Frontend JS: Listened to the song. Asking for confirmation to save the song.');
                alert("Ended listening.");
                //reset the speech recongition after we have stopped listening to the song
                speech.stop();
                transcription.innerHTML = "";
                interim_span.innerHTML = "";
                speech.start();
                askForConfirmation();
                ContinueListening = false;
                stopListening = false;
            }
        }
        await sleep(gap);
    }
}

function analyserCallback(result) {
    //we update the note which is a global variable
    if (micVol >= minMicVol) {
        detectedNote = result.note;
    }
}

function sendSongToBackend(data) {
    outputData.songOutgoingData.data = data;
}

function askForConfirmation() {
    outputData.askSingingConfirmation.data = "confirm song?"
}