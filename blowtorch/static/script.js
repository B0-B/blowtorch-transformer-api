const { createApp, ref } = Vue

createApp({
    data () {
        return {
            message: 'Hello Vue!',
            count: 0
        }
    },
    methods: {
        incrementCount() {
            this.count++
        },
        messageBox (msg, side) {

            const chatWindow = document.getElementById('chat-window');
            
            // generate message box wrapper
            const boxWrapper = document.createElement('span');
            boxWrapper.classList.add('message-box-wrapper');
            
            // generate message box
            const messageBox = document.createElement('span');
            messageBox.classList.add('message-box');
            
            // decide which side
            if (side == 'l') {
                messageBox.classList.add('left')
            } else if (side == 'r') {
                messageBox.classList.add('right')
            }

            // add content
            messageBox.innerHTML = msg;
            
            boxWrapper.appendChild(messageBox);
            chatWindow.appendChild(boxWrapper)
            
        },
        request (options, path, json=true) {
            return new Promise(function (resolve, reject) {
                var xhr = new XMLHttpRequest(); 
                xhr.open("POST", path, true); 
                if (json) {
                    xhr.setRequestHeader("Content-type", "application/json;charset=UTF-8"); 
                }
                xhr.onreadystatechange = function () {  
                    if (xhr.readyState == 4 && xhr.status == 200) {
                        var json = JSON.parse(xhr.responseText);
                        if (Object.keys(json).includes('errors') && json['errors'].length != 0) { // if errors occur
                            console.log('server:', json['errors'])
                        } resolve(json);
                    }
                }
                xhr.onerror = function(e) {
                    reject({'errors': ['error during request: no connection']})
                }
                xhr.send(JSON.stringify(options)); 
            });
        },

    },
    mounted () {
        try {
            // this.request({'data':[1,2,3]}, '/');
            console.log('mounted.');
            this.messageBox('This is a test', 'l');
            this.messageBox('This is another test', 'r');
            
        } catch (error) {
            console.log('mount hook error:', error)
        }
    }
}).mount('#vue-app')