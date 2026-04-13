import React from 'react';
import {Box, Text, useInput} from 'ink';

const UI_DEBUG = (process.env.NODE_UI_DEBUG || '0').toLowerCase() !== '0';

function debugUI(event, payload = {}) {
  if (!UI_DEBUG) return;
  console.error(`[node-ui][input] ${event}`, payload);
}

function TextInput({value, placeholder, busy, onChange, onSubmit, onExit, onDebug}) {
  const [cursorVisible, setCursorVisible] = React.useState(true);

  const reportDebug = React.useCallback((event, payload = {}) => {
    debugUI(event, payload);
    if (typeof onDebug === 'function') onDebug('input', event, payload);
  }, [onDebug]);

  React.useEffect(() => {
    const timer = setInterval(() => setCursorVisible(prev => !prev), 500);
    return () => clearInterval(timer);
  }, []);

  useInput((input, key) => {
    reportDebug('key', {input, key});
    if (key.ctrl && input === 'c') {
      reportDebug('exit:ctrl_c');
      if (onExit) onExit();
      return;
    }
    if (typeof input === 'string' && /.+[\r\n]+/.test(input)) {
      const sanitized = input.replace(/\r/g, '\n');
      const [beforeSubmit] = sanitized.split('\n');
      const nextValue = `${value}${beforeSubmit}`.trim();
      reportDebug('submit:embedded_newline', {input, beforeSubmit, nextValue});
      if (beforeSubmit && onChange) onChange(value + beforeSubmit);
      if (onSubmit) onSubmit(nextValue);
      return;
    }
    if (key.return || input === '\r' || input === '\n') {
      reportDebug('submit:enter', {value});
      if (onSubmit) onSubmit(value.trim());
      return;
    }
    if (key.backspace || key.delete) {
      reportDebug('edit:backspace', {previous: value});
      if (onChange) onChange(value.slice(0, -1));
      return;
    }
    if (input && !key.ctrl && !key.meta) {
      reportDebug('edit:append', {input, previous: value});
      if (onChange) onChange(value + input);
    }
  }, {isActive: process.stdin.isTTY === true});

  const display = value.length > 0 ? value : placeholder;
  const valueColor = value.length > 0 ? 'white' : 'gray';
  const promptColor = value.length > 0 ? 'cyanBright' : 'gray';
  const cursor = cursorVisible ? '▍' : ' ';

  return React.createElement(Box, {
      borderStyle: 'round',
      borderColor: 'cyanBright',
      backgroundColor: '#0f172a',
      paddingX: 1,
      paddingY: 0,
    },
    React.createElement(Text, {color: promptColor, bold: true}, '❯ '),
    React.createElement(Text, {color: valueColor}, display),
    React.createElement(Text, {color: 'cyanBright', bold: true}, `${busy ? ' …' : ''}${cursor}`)
  );
}

export {TextInput};
