# Repilot Patch

```
      if (nameNode.getFirstChild().getType() == Token.NAME) {
        String name = nameNode.getFirstChild().getString();
        if (name.equals("Math")) {
          return false;
        }
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

@@ -955,6 +955,12 @@

       }
 
       // Functions in the "Math" namespace have no side effects.
+      if (nameNode.getFirstChild().getType() == Token.NAME) {
+        String name = nameNode.getFirstChild().getString();
+        if (name.equals("Math")) {
+          return false;
+        }
+      }
 
       if (compiler != null && !compiler.hasRegExpGlobalReferences()) {
         if (nameNode.getFirstChild().getType() == Token.REGEXP
```

# Note

