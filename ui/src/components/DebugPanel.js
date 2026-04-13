import React from 'react';
import {Box, Text} from 'ink';

function DebugPanel({lines, copyStatus}) {
  const items = Array.isArray(lines) ? lines.slice(-8) : [];

  return React.createElement(
    Box,
    {
      marginTop: 1,
      flexDirection: 'column',
      borderStyle: 'round',
      borderColor: 'yellow',
      paddingX: 1,
      paddingY: 0,
    },
    React.createElement(Text, {color: 'yellow', bold: true}, 'debug node-ui'),
    React.createElement(
      Text,
      {color: copyStatus?.startsWith('copiado') ? 'green' : 'gray'},
      copyStatus || '📋 copiar todo: Ctrl+Y'
    ),
    items.length > 0
      ? items.map((line, index) => React.createElement(Text, {key: `${index}-${line}`, color: 'gray'}, line))
      : React.createElement(Text, {color: 'gray'}, 'sin eventos todavía')
  );
}

export {DebugPanel};
