import React from 'react';
import {Box, Text, useApp, useInput} from 'ink';
import {TextInput} from '../components/TextInput.js';
import {TranscriptList} from '../components/TranscriptList.js';
import {useBridgeSession} from '../hooks/useBridgeSession.js';

function REPL() {
  const {exit: inkExit} = useApp();
  const {state, submit, exit} = useBridgeSession({onExit: inkExit});
  const [prompt, setPrompt] = React.useState('');
  const [scrollOffset, setScrollOffset] = React.useState(0);
  const viewportHeight = Math.max((process.stdout.rows || 24) - 12, 8);
  const maxScroll = Math.max(0, state.transcript.length - viewportHeight);

  React.useEffect(() => {
    setScrollOffset(prev => Math.min(prev, maxScroll));
  }, [maxScroll]);

  useInput((input, key) => {
    if (key.upArrow) {
      setScrollOffset(prev => Math.min(prev + 1, maxScroll));
      return;
    }
    if (key.downArrow) {
      setScrollOffset(prev => Math.max(prev - 1, 0));
      return;
    }
    if (key.pageUp) {
      setScrollOffset(prev => Math.min(prev + Math.max(4, Math.floor(viewportHeight / 2)), maxScroll));
      return;
    }
    if (key.pageDown) {
      setScrollOffset(prev => Math.max(prev - Math.max(4, Math.floor(viewportHeight / 2)), 0));
      return;
    }
    if (key.home) {
      setScrollOffset(maxScroll);
      return;
    }
    if (key.end) {
      setScrollOffset(0);
    }
  }, {isActive: process.stdin.isTTY === true});

  const handleSubmit = React.useCallback(value => {
    const text = value.trim();
    if (!text) return;
    if (text.toLowerCase() === 'salir' || text.toLowerCase() === 'exit' || text.toLowerCase() === 'quit') {
      setPrompt('');
      exit();
      inkExit();
      return;
    }
    if (state.busy) {
      return;
    }
    setPrompt('');
    submit(text);
  }, [exit, inkExit, submit, state.busy]);

  const lines = state.transcript;

  return React.createElement(Box, {flexDirection: 'column'},
    React.createElement(Box, {borderStyle: 'round', borderColor: 'cyan', paddingX: 1, paddingY: 0, flexDirection: 'column'},
      React.createElement(Text, {color: 'cyan', bold: true}, 'Multi-Agentes'),
      React.createElement(Text, {color: 'gray'}, state.status),
      React.createElement(Text, {color: 'gray'}, state.session_id ? `session ${state.session_id}` : 'sesión…')
    ),
    React.createElement(Box, {marginTop: 1, flexDirection: 'column', flexGrow: 1},
      React.createElement(TranscriptList, {lines, scrollOffset, height: viewportHeight})
    ),
    React.createElement(TextInput, {
      value: prompt,
      placeholder: state.prompt || 'Escribí un mensaje, /help o salir',
      busy: state.busy,
      onChange: setPrompt,
      onSubmit: handleSubmit,
      onExit: () => {
        exit();
        inkExit();
      },
    }),
    React.createElement(Text, {color: 'gray'}, '↑↓ scroll · PgUp/PgDn pagina · End abajo · Home arriba')
  );
}

export {REPL};
