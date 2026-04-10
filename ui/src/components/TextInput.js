import React from 'react';
import {Box, Text, useInput} from 'ink';

function TextInput({value, placeholder, busy, onChange, onSubmit, onExit}) {
  const [cursorVisible, setCursorVisible] = React.useState(true);

  React.useEffect(() => {
    const timer = setInterval(() => setCursorVisible(prev => !prev), 500);
    return () => clearInterval(timer);
  }, []);

  useInput((input, key) => {
    if (key.ctrl && input === 'c') {
      if (onExit) onExit();
      return;
    }
    if (key.return) {
      if (onSubmit) onSubmit(value.trim());
      return;
    }
    if (key.backspace || key.delete) {
      if (onChange) onChange(value.slice(0, -1));
      return;
    }
    if (input && !key.ctrl && !key.meta) {
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
