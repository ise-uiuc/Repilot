# Repilot Patch

```
private static String nullString = "\0";
```

# Developer Patch

```
    private static String nullString = String.valueOf('\u0000');
```

# Context

```
--- bug/Jsoup-17/src/main/java/org/jsoup/parser/TreeBuilderState.java

+++ fix/Jsoup-17/src/main/java/org/jsoup/parser/TreeBuilderState.java

@@ -1448,7 +1448,7 @@

         }
     };
 
-    private static String nullString = String.valueOf(0x0000);
+private static String nullString = "\0";
 
     abstract boolean process(Token t, TreeBuilder tb);
```

# Note

