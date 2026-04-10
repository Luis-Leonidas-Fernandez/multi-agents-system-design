import React from 'react';
import {Box} from 'ink';

function App({children}) {
  return React.createElement(Box, {flexDirection: 'column', paddingX: 1, paddingY: 0}, children);
}

export {App};
