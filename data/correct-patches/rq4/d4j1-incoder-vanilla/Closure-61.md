# Repilot Patch

```
      if (nameNode.getFirstChild().getType() == Token.NAME
          && "Math".equals(nameNode.getFirstChild().getString())) {
        return false;
      }
```

# Developer Patch

```
      if (nameNode.getFirstChild().getType() == Token.NAME) {
        String namespaceName = nameNode.getFirstChild().getString();
        if (namespaceName.equals("Math")) {
          return false;
        }
      }
```

# Context

```
--- bug/Closure-61/src/com/google/javascript/jscomp/NodeUtil.java

+++ fix/Closure-61/src/com/google/javascript/jscomp/NodeUtil.java

@@ -955,6 +955,10 @@

       }
 
       // Functions in the "Math" namespace have no side effects.
+      if (nameNode.getFirstChild().getType() == Token.NAME
+          && "Math".equals(nameNode.getFirstChild().getString())) {
+        return false;
+      }
 
       if (compiler != null && !compiler.hasRegExpGlobalReferences()) {
         if (nameNode.getFirstChild().getType() == Token.REGEXP
```

# Note

