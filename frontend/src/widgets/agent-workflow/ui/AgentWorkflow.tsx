type Props = {
  reasoning: string
  conclusion: string
  finalResponse: string
}

export function AgentWorkflow({ reasoning, conclusion, finalResponse }: Props) {
  return (
    <section>
      <h2>Reasoning</h2>
      <p>{reasoning}</p>
      <h2>Conclusion</h2>
      <p>{conclusion}</p>
      <h2>Final response</h2>
      <p>{finalResponse}</p>
    </section>
  )
}
