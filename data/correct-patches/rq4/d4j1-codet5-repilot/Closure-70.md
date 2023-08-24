# Repilot Patch

```
jsDocParameter.getJSType(), false);
```

# Developer Patch

```
jsDocParameter.getJSType(), false);
```

# Context

```
--- bug/Closure-70/src/com/google/javascript/jscomp/TypedScopeCreator.java

+++ fix/Closure-70/src/com/google/javascript/jscomp/TypedScopeCreator.java

@@ -1742,7 +1742,7 @@

           for (Node astParameter : astParameters.children()) {
             if (jsDocParameter != null) {
               defineSlot(astParameter, functionNode,
-                  jsDocParameter.getJSType(), true);
+ jsDocParameter.getJSType(), false);
               jsDocParameter = jsDocParameter.getNext();
             } else {
               defineSlot(astParameter, functionNode, null, true);
```

# Note

