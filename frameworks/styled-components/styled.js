// Styled Components uses Stylis for CSS preprocessing
const { serialize, compile, stringify } = stylis;
// Used to generate a unique ID per styled component
let counter = 0;
// Reference to the <style> tag injected in the <head>
let styleTag = null;

/**
 * Simple hashing function, the real styled-components uses the MurmurHash
 * algorithm and then converts the hash number to an alphabetic name:
 *
 * https://github.com/styled-components/styled-components/blob/v3.3.3/src/utils/generateAlphabeticName.js#L13
 *
 * @param {String} s
 * @returns {Number}
 */
const hash = (s) =>
  s.split("").reduce((a, b) => {
    a = (a << 5) - a + b.charCodeAt(0);
    return a & a;
  }, 0);

/**
 * Styled Components generates its own <style> tag which contain all the
 * styles for each component. If the tag doesn't exist, this function will
 * create and inject it into the head
 */
const createOrGetStyleTag = () => {
  if (!styleTag) {
    const style = document.createElement("style");
    style.setAttribute("data-styled-components", "");
    document.head.appendChild(style);
    styleTag = style;
  }
  return styleTag;
};

/**
 * Concat all the string chunks from the tagged template[1] with the
 * results of the expressions one by one; and if an expression is a
 * function it is called with the component’s props.
 *
 * @param {String[]} - Array of template strings
 * @param {Function[]} - Array of template expressions
 * @param {Record} - Component props
 */
const interpolateStyles = (strs, exprs, props) => {
  const evaluatedStyles = exprs.reduce((result, expr, index) => {
    const isFunc = typeof expr === "function";
    const value = isFunc ? expr(props) : expr;
    return result + value + strs[index + 1];
  }, strs[0]);

  return evaluatedStyles;
};

/**
 * Classnames are unique based off the componentId and evaluated styles.
 * Which means if we evaluate to new styles we create a new classname. Unused
 * styles aren't removed because of unecessary performance overhead:
 *
 * https://github.com/styled-components/styled-components/issues/1431#issuecomment-358097912
 */
const createClassname = (componentId, evaluatedStyles) => {
  const className = "c-" + hash(componentId + evaluatedStyles);
  return className;
};

/**
 * Inject the styles into the <style> tag if the className doesn't already
 * exist
 */
const injectStyles = (className, evaluatedStyles) => {
  const selector = "." + className;
  const style = createOrGetStyleTag();
  if (!style.innerText.includes(selector)) {
    const cssStr = serialize(
      compile(`${selector}{${evaluatedStyles}}`),
      stringify
    );
    style.appendChild(document.createTextNode(cssStr));
  }
};

const styled = (TargetComponent) => (strs, ...exprs) => {
  counter++;
  const componentId = "sc-" + hash("sc" + counter);

  const StyledComponent = ({ className = "", ...props }) => {
    // On every rerender we evaluate the styles since props has changed
    const evaluatedStyles = interpolateStyles(strs, exprs, props);
    // Create a className based off newly evaluated styles
    const generatedClassName = createClassname(componentId, evaluatedStyles);

    // We only inject in new styles if there are changes to the newly
    // evaluated styles
    React.useEffect(() => {
      injectStyles(generatedClassName, evaluatedStyles);
    }, [generatedClassName]);

    return (
      <TargetComponent
        {...props}
        className={className + " " + componentId + " " + generatedClassName}
      />
    );
  };

  StyledComponent.componentId = componentId;

  return StyledComponent;
};

styled.button = styled("button");
styled.span = styled("span");

window.styled = styled;

/**
[1]
A tagged template:

const Button = styled.button`
  color: coral;
  background: ${(props) => props.success ? 'green' : 'blue'};
  padding: 1rem;
`;

Is the same as:

const Button = styled('button')([
  'color: coral; background: ',
  '; padding: 0.25rem 1rem;'
], (props) => props.success ? 'green' : 'blue');

https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Template_literals#Tagged_templates
*/
